"""19-call classical diagnostic through survey/analyze/submit, with exact replay.

9 survey calls + 9 sandboxed analyses + final submission. Each request contains
one tool, so decision count is conservative relative to batched provider calls.
No truth, generator files, unmeasured water data, or paid provider in the policy.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from laketrace.data.store import DataStore
from laketrace.investigation.session import Session
from laketrace.util import stable_hash

POLICY=(ROOT/'laketrace/baselines/field_reference.py').read_text()
FOOTER='''
nav=np.array([[r[k] for k in 'xyz'] for r in data['navigation']])
rows=data['target_nodes']
obs=np.array([[r[k] for k in ['x','y','z','target_value','u','v','w']] for r in rows])
r=infer(nav,obs,HORIZON)
measured={int(r['node_id']) for r in rows}
remaining=np.array([i for i in range(len(nav)) if i not in measured])
points=[]
if len(remaining):
 source=np.array([r['source_position'][k] for k in 'xyz'])
 distance=np.linalg.norm((nav[remaining]-source)/np.maximum(np.ptp(nav,axis=0),1),axis=1)
 local=remaining[np.argsort(distance)[:3]]
 chosen=list(local)
 scaled=nav/np.maximum(np.ptp(nav,axis=0),1)
 tree=cKDTree(obs[:,:3]/np.maximum(np.ptp(nav,axis=0),1))
 novelty=tree.query(scaled[remaining])[0]
 for j in range(3):
  selected=int(remaining[np.argmax(novelty)]);chosen.append(selected)
  novelty=np.minimum(novelty,np.linalg.norm(scaled[remaining]-scaled[selected],axis=1))
 points=[dict(zip('xyz',nav[i].tolist())) for i in dict.fromkeys(chosen)]
r['next_points']=points
r['plume_node_id']=int(max(rows,key=lambda s:s['target_value']-s['background_median'])['node_id'])
print(__import__('json').dumps(r))
'''

def episode(root,t,out):
 out.mkdir(parents=True,exist_ok=False)
 protocol=json.loads((ROOT/'configs/investigation/three_waterbody_v22_survey_v6.json').read_text());store=DataStore(root)
 session=Session(store,t['instance_id'],0,protocol,{'target_channel':'fdom'},workspace=out/'workbench')
 actions=[];fits=[];admission=None;begin=time.monotonic()
 def act(name,args):
  session.env.provider_call_count=len(actions)+1
  value=session.execute(name,args)
  actions.append({'name':name,'args':args,'ok':value.get('ok'),'result':value})
  if not value.get('ok'):raise RuntimeError(str(value)[:1500])
  return value
 code=POLICY+'\nHORIZON='+repr(t['observation_contract']['snapshot_time_s'])+'\n'+FOOTER
 try:
  points=None
  for turn in range(9):
   act('survey',{'pattern':'coarse_3d','max_actions':12,'fit_after':False} if points is None else {'points':points,'max_actions':12,'fit_after':False})
   result=act('analyze',{'code':code})
   fit=json.loads(result['stdout'].strip());fits.append(fit);points=fit['next_points']
   if turn==7:
    from laketrace.investigation.runner import prepare_request
    config=json.loads((ROOT/'configs/investigation/models/mimo_v26_flash_tokenplan_v22_survey_v6_budget20.json').read_text())
    body,error,_=prepare_request(session,config,protocol,session.context(),[],4,0.)
    admission={'error':error,'request_bytes':len(json.dumps(body).encode()) if body else 0,'measured_nodes':len(session.env.measurements),'scope':'Full accumulated grid, four calls remaining; no provider request sent.'}
    assert error is None,error
   if not points:break
  report={'source_present':None,'source_presence_probability':.5,'mechanism':None,'plume_node_id':fits[-1]['plume_node_id'],'source_position':fits[-1]['source_position'],'source_confidence':0.,'decision':'cannot_determine','explanation':'Approximate ADE localization reference. L1 and L3 are outside its scope; probability fields are interface placeholders.'}
  act('submit_report',report)
  assert session.final and len(actions)<=20
  ledger=stable_hash(session.env.sample_ledger);events=stable_hash(session.env.events)
  summary={'instance_id':t['instance_id'],'calls':len(actions),'movement_actions':session.env.successful_action_count,'samples':len(session.env.sample_ledger),'unique_nodes':len(session.env.measurements),'report':session.report,'ledger_sha256':ledger,'elapsed_s':time.monotonic()-begin,'context_admission':admission}
 finally:session.close()
 (out/'actions.json').write_text(json.dumps(actions,indent=2));(out/'fits.json').write_text(json.dumps(fits,indent=2))
 replay=Session(store,t['instance_id'],0,protocol,{'target_channel':'fdom'},workspace=out/'workbench',replay=True)
 try:
  for i,a in enumerate(actions):
   replay.env.provider_call_count=i+1;value=replay.execute(a['name'],a['args'])
   assert value.get('ok'),value
   if a['name']=='analyze':assert value['stdout']==a['result']['stdout']
  assert stable_hash(replay.env.sample_ledger)==ledger and stable_hash(replay.env.events)==events
  assert replay.report==summary['report']
  summary['replay_pass']=True
 finally:replay.close()
 (out/'episode.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);p.add_argument('--limit',type=int);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 tasks=DataStore(a.root).tasks()[:a.limit]
 for t in tasks:
  r=episode(a.root,t,a.out/t['instance_id']);print(json.dumps(r),flush=True)
