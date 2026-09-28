"""Registered observation bias/drift; hidden realizations never enter the context."""
import math
from laketrace.env.v20_3 import EnvironmentV20_3
from .temporal import TemporalEnvironment


class BiasMixin:
    def _provider_round_available(self):
        if getattr(self, "unlimited_provider_calls", False):
            return True
        if getattr(self, 'separate_provider_actions', False):
            return self.provider_call_count < self.max_turns
        return super()._provider_round_available()

    def _counter_gauss(self, sigma, *parts):
        value = super()._counter_gauss(sigma, *parts)
        model = self.task.get('observation_contract', {}).get('sensor_uncertainty', {})
        if parts and parts[0] == 'channel' and model:
            channel = parts[3]
            lod = self.limits[(self.task['waterbody'], channel, 'surface')][1]
            bias = super()._counter_gauss(lod*model.get('persistent_bias_lod', 0.), 'persistent_bias', channel)
            interval = model.get('drift_interval_s', 200.)
            position = self.budget.used_s/interval
            left = math.floor(position)
            a = super()._counter_gauss(lod*model.get('drift_lod', 0.), 'drift', channel, left)
            b = super()._counter_gauss(lod*model.get('drift_lod', 0.), 'drift', channel, left+1)
            value += bias+a+(position-left)*(b-a)
        return value


class MeasuredEnvironment(BiasMixin, EnvironmentV20_3):
    pass


class ClockedMeasuredEnvironment(BiasMixin, TemporalEnvironment):
    pass
