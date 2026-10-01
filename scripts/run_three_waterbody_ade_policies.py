#!/usr/bin/env python3
"""Run systematic and adaptive ADE localization through the survey-v6 cockpit.

The two policies share the same observation-only inversion. The adaptive policy
chooses new waypoints by approximate predictive disagreement among fitted ADE
source alternatives. Truth is opened only after replay.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from laketrace.data.store import DataStore
from laketrace.investigation.session import Session
from laketrace.scoring.scorer import score_submission
from laketrace.util import stable_hash

INVERSION = (ROOT / "laketrace/baselines/field_reference.py").read_text()
ANALYSIS = r'''
nav=np.array([[row[a] for a in 'xyz'] for row in data['navigation']],float)
rows=data['target_nodes']
obs=np.array([[row[k] for k in ['x','y','z','target_value','u','v','w']] for row in rows],float)
r=infer(nav,obs,HORIZON)
r['plume_node_id']=int(max(rows,key=lambda s:(s['target_value']-s['background_median'],-s['node_id']))['node_id'])
measured={int(s['node_id']) for s in rows}
candidate=np.array([i for i in spatial_candidates(nav,256) if int(i) not in measured],int)
next_points=[]
systematic_points=[]
if len(candidate):
    alternatives=r['alternatives']
    sources=np.array([item['position'] for item in alternatives],float)
    spacing=np.array([np.min(np.diff(np.unique(nav[:,i]))) if len(np.unique(nav[:,i]))>1 else 1. for i in range(3)])
    predictions=kernel(nav[candidate],sources,np.array(r['velocity']),r['diffusivity'],r['duration_s'],spacing)*r['amplitude']
    rss=np.array([item['train_rss'] for item in alternatives])
    variance=max(r['source_train_rss']/max(len(obs)-9,1),1e-8)
    weights=np.exp(-np.minimum((rss-rss.min())/(2*variance),100.));weights/=weights.sum()
    predicted_mean=predictions@weights
    disagreement=((predictions-predicted_mean[:,None])**2)@weights
    scale=np.maximum(np.ptp(nav,axis=0),1.)
    scaled=nav/scale
    tree=cKDTree(obs[:,:3]/scale)
    novelty=tree.query(scaled[candidate])[0]
    current=obs[-1,:3]/scale
    coverage=novelty.copy()
    coverage_current=current.copy()
    coverage_chosen=[]
    for _ in range(min(6,len(candidate))):
        travel=np.linalg.norm(scaled[candidate]-coverage_current,axis=1)
        utility=coverage/(1.+travel)
        utility[[j for j,i in enumerate(candidate) if int(i) in coverage_chosen]]=-1.
        selected=int(candidate[np.argmax(utility)]);coverage_chosen.append(selected)
        coverage_current=scaled[selected]
        coverage=np.minimum(coverage,np.linalg.norm(scaled[candidate]-coverage_current,axis=1))
    systematic_points=[dict(zip('xyz',nav[i].tolist())) for i in coverage_chosen]
    chosen=[]
    for _ in range(min(6,len(candidate))):
        travel=np.linalg.norm(scaled[candidate]-current,axis=1)
        gain=.5*np.log1p(disagreement/variance)
        utility=(gain+.03*novelty)*(0.5+np.minimum(novelty,1.))/(1.+travel)
        utility[[j for j,i in enumerate(candidate) if int(i) in chosen]]=-1.
        selected=int(candidate[np.argmax(utility)]);chosen.append(selected)
        current=scaled[selected]
        novelty=np.minimum(novelty,np.linalg.norm(scaled[candidate]-current,axis=1))
    next_points=[dict(zip('xyz',nav[i].tolist())) for i in chosen]
r['next_points']=next_points
r['systematic_points']=systematic_points
r['acquisition']='approximate ADE alternative predictive disagreement with novelty and travel cost'
print(__import__('json').dumps(r))
'''


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def run_episode(store, task, seed, policy, protocol, directory):
    directory.mkdir(parents=True, exist_ok=False)
    session = Session(store, task['instance_id'], seed, protocol,
                      {'target_channel': 'fdom'}, workspace=directory/'workbench')
    actions, fits = [], []
    begin = time.monotonic()

    def act(name, args):
        session.env.provider_call_count = len(actions) + 1
        value = session.execute(name, args)
        actions.append({'name': name, 'args': args, 'ok': value.get('ok'),
                        'result_sha256': stable_hash(value),
                        **({'stdout': value['stdout']} if name == 'analyze' and value.get('ok') else {})})
        if not value.get('ok'):
            raise RuntimeError(f"{task['instance_id']} {policy} seed {seed}: {name}: {value}")
        return value

    code = INVERSION + '\nHORIZON=' + repr(task['observation_contract']['snapshot_time_s']) + '\n' + ANALYSIS
    try:
        points = None
        for _ in range(9):
            survey_args = {'pattern':'coarse_3d','max_actions':12,'fit_after':False}
            if points:
                survey_args = {'points':points,'max_actions':12,'fit_after':False}
            act('survey', survey_args)
            result = act('analyze', {'code':code})
            fit = json.loads(result['stdout'].strip())
            fits.append(fit)
            points = fit['next_points' if policy == 'adaptive' else 'systematic_points']
        report = {'source_present':None, 'source_presence_probability':0.5,
                  'mechanism':None, 'plume_node_id':fits[-1]['plume_node_id'],
                  'source_position':fits[-1]['source_position'],
                  'source_confidence':0.0, 'decision':'cannot_determine',
                  'explanation':'Approximate ADE localization reference; L1 and L3 are outside scope.'}
        act('submit_report', report)
        assert session.final and len(actions) == 19
        ledger_hash = stable_hash(session.env.sample_ledger)
        event_hash = stable_hash(session.env.events)
        answer = session.canonical_answer()
        outcome = {'instance_id':task['instance_id'], 'policy':policy, 'seed':seed,
                   'status':'completed', 'calls':len(actions),
                   'movement_actions':session.env.successful_action_count,
                   'samples':len(session.env.sample_ledger),
                   'unique_nodes':len(session.env.measurements),
                   'distance_m':session.env.budget.used_m, 'report':session.report,
                   'answer':answer, 'ledger_sha256':ledger_hash,
                   'events_sha256':event_hash, 'elapsed_s':time.monotonic()-begin}
    finally:
        session.close()
    save(directory/'actions.json', actions)
    save(directory/'fits.json', fits)

    replay = Session(store,task['instance_id'],seed,protocol,{'target_channel':'fdom'},
                     workspace=directory/'workbench',replay=True)
    try:
        for i, action in enumerate(actions):
            replay.env.provider_call_count = i+1
            value = replay.execute(action['name'], action['args'])
            assert stable_hash(value) == action['result_sha256'], (task['instance_id'],policy,seed,i,'result')
            if action['name'] == 'analyze':
                assert value['stdout'] == action['stdout']
        assert stable_hash(replay.env.sample_ledger) == ledger_hash
        assert stable_hash(replay.env.events) == event_hash
        assert replay.report == outcome['report']
        outcome['replay_pass'] = True
    finally:
        replay.close()
    save(directory/'episode.json', outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--instances',nargs='+',help='Run these task IDs, in manifest order')
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    data_root = ROOT/'data/three_waterbody_candidate_v22'
    protocol_path = ROOT/'configs/investigation/three_waterbody_v22_survey_v6.json'
    protocol = json.loads(protocol_path.read_text())
    store = DataStore(data_root)
    tasks = store.tasks()
    if args.instances:
        selected = set(args.instances)
        if len(selected) != len(args.instances) or not selected <= {t['instance_id'] for t in tasks}:
            parser.error('--instances requires unique known task IDs')
        tasks = tuple(t for t in tasks if t['instance_id'] in selected)
    tasks = tasks[:args.limit]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    save(output/'manifest.json', {
        'data_root':str(data_root), 'protocol_id':protocol['id'],
        'task_count':len(tasks), 'seeds':args.seeds,
        'policies':['systematic','adaptive'],
        'algorithm':'bounded-memory approximate ADE point-source inversion plus fitted quadratic background',
        'adaptive_acquisition':'predictive disagreement among ADE source alternatives, novelty and travel cost',
        'scope':'L2 and L4 localization only; L1/L3 placeholders are excluded',
        'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__),ROOT/'laketrace/baselines/field_reference.py',protocol_path,
                            data_root/'tasks/instances.jsonl',data_root/'truth/exam_truth.jsonl']}})
    episodes=[]
    for policy in ('systematic','adaptive'):
        for seed in args.seeds:
            for index, task in enumerate(tasks,1):
                outcome = run_episode(store,task,seed,policy,protocol,
                                      output/policy/f'seed{seed}'/task['instance_id'])
                episodes.append(outcome)
                print(json.dumps({'policy':policy,'seed':seed,'completed':index,'of':len(tasks),
                                  'instance_id':task['instance_id'],'calls':outcome['calls'],
                                  'samples':outcome['samples'],'replay':outcome['replay_pass'],
                                  'elapsed_s':round(outcome['elapsed_s'],1)}),flush=True)
    evaluator = DataStore(data_root,evaluator=True)
    rows=[]
    for episode in episodes:
        truth = evaluator.truth(episode['instance_id'])
        score = score_submission(episode['answer'],truth,evaluator.protocol['tolerances'][truth['waterbody']])
        rows.append({'instance_id':episode['instance_id'],'policy':episode['policy'],'seed':episode['seed'],
                     'waterbody':truth['waterbody'],'source_present':truth['detectable_source'],
                     'mechanism':truth['L3_mechanism'], 'L2_pass':score['L2_pass'],
                     'L4_pass':score['L4_pass'],
                     'L4_horizontal_error_m':score['L4_horizontal_error_m'],
                     'L4_vertical_error_m':score['L4_vertical_error_m'],
                     'calls':episode['calls'],'samples':episode['samples'],
                     'unique_nodes':episode['unique_nodes'],'distance_m':episode['distance_m'],
                     'replay_pass':episode['replay_pass']})
    with (output/'scores.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={'episodes':len(rows),'replay_pass':all(r['replay_pass'] for r in rows),'policies':{}}
    for policy in ('systematic','adaptive'):
        selected=[r for r in rows if r['policy']==policy]
        positive=[r for r in selected if r['source_present']]
        summary['policies'][policy]={'episodes':len(selected),'positive':len(positive),
            'L2_correct':sum(r['L2_pass'] is True for r in positive),
            'L4_correct':sum(r['L4_pass'] is True for r in positive),
            'L4_rate':sum(r['L4_pass'] is True for r in positive)/len(positive) if positive else None,
            'mean_samples':sum(r['samples'] for r in selected)/len(selected),
            'mean_distance_m':sum(r['distance_m'] for r in selected)/len(selected)}
    save(output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
