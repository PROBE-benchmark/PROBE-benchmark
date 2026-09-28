"""Optional clocked observations; hidden frames never enter analysis snapshots."""
from dataclasses import replace
import math

from laketrace.env.v20_3 import EnvironmentV20_3


class TemporalEnvironment(EnvironmentV20_3):
    def __init__(self, store, instance_id, seed, **options):
        self.horizon_only = options.pop('observation_horizon_only', False)
        contract = store.task(instance_id)['observation_contract']
        self.frame_times = contract['frame_times_s']
        if (len(self.frame_times)<2 or self.frame_times[0]!=0 or
            any(not math.isfinite(t) for t in self.frame_times) or
            any(b<=a for a,b in zip(self.frame_times,self.frame_times[1:]))):
            raise ValueError('dynamic frames require increasing mission times starting at zero')
        self.temporal_frames = [store.frame(instance_id,f'F{i+1}') for i in range(len(self.frame_times))]
        geometry = tuple(n.xyz for n in self.temporal_frames[0].nodes)
        if any(tuple(n.xyz for n in frame.nodes)!=geometry for frame in self.temporal_frames):
            raise ValueError('temporal frames must share the exact navigation grid')
        if not self.horizon_only and store.task(instance_id)['budget']['endurance_s'] > self.frame_times[-1]:
            raise ValueError('dynamic observations do not span the mission budget')
        super().__init__(store,instance_id,seed,**options)

    def apply_resource_stage(self):
        super().apply_resource_stage()
        if self.horizon_only and self.budget.used_s >= self.frame_times[-1]:
            self.enter_submit_only('observation_horizon')

    def _maybe_submit_only(self):
        if self.horizon_only:
            # A long edge may be unavailable while a shorter edge or wait fits.
            self.apply_resource_stage()
        else:
            super()._maybe_submit_only()

    def _edge_affordable(self, edge):
        if self.horizon_only:
            return self.budget.used_s + self._edge_time(edge) <= self.frame_times[-1]
        return super()._edge_affordable(edge)

    def move(self, directions, **kwargs):
        result = super().move(directions, **kwargs)
        if self.horizon_only and result.get('code') == 'E_PHYSICAL_BUDGET':
            result['code'] = 'E_OBSERVATION_HORIZON'
            result['detail'] = 'Next edge would leave registered temporal data coverage; existing evidence remains usable.'
            for key in ('route_stop_reason', 'route_intervention_event', 'next_inference_event'):
                if result.get(key) == 'physical_budget_unaffordable':
                    result[key] = 'observation_horizon'
        return result

    def _node(self, index):
        time = self.budget.used_s
        if not 0 <= time <= self.frame_times[-1] + 1e-6:
            raise ValueError('observation outside registered time coverage')
        upper = next((i for i,t in enumerate(self.frame_times) if t>=time),len(self.frame_times)-1)
        if upper==0:
            return self.temporal_frames[0].by_xyz[self.graph.positions[index]]
        lower=upper-1
        fraction=(time-self.frame_times[lower])/(self.frame_times[upper]-self.frame_times[lower])
        a,b=self.temporal_frames[lower].by_xyz[self.graph.positions[index]],self.temporal_frames[upper].by_xyz[self.graph.positions[index]]
        return replace(a,channels={c:a.channels[c]+fraction*(b.channels[c]-a.channels[c]) for c in a.channels},
                       flow={c:a.flow[c]+fraction*(b.flow[c]-a.flow[c]) for c in a.flow})

    def wait_and_measure(self, seconds):
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 1<=seconds<=300:
            raise ValueError('wait duration must be 1-300 seconds')
        remaining = self.frame_times[-1]-self.budget.used_s if self.horizon_only else self.budget.remaining_s
        if self.mode=='submit_only' or seconds>remaining:
            raise ValueError('registered observation horizon exhausted' if self.horizon_only else 'observation time budget exhausted')
        self.budget.used_s+=seconds
        self._successful_action_count+=1
        reading=self._measure(self.current,mode='transit')
        event={'ok':True,'action':'wait','seconds':seconds,'mission_time_s':self.budget.used_s,'reading':reading}
        self.events.append(event)
        self.apply_resource_stage()
        return event
