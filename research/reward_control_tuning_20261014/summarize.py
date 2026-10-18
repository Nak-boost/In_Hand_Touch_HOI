from pathlib import Path
import json,numpy as np
out=Path(__file__).parent
result={}
def stats(x):
 x=np.asarray(x);return {'mean':float(x.mean()),'p50':float(np.quantile(x,.5)),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}
for scene in ['inhand','tabletop']:
 d=np.load(out/(scene+'_samples.npz'));meta=json.loads((out/(scene+'_metadata.json')).read_text());gamma=meta['gamma'];cfg=meta['config']['task']['env']['manipulation'];scene_report={}
 for protocol in ['hold','small_noise','outward_push','alternating']:
  v={k.split('__')[1]:d[k] for k in d.files if k.startswith(protocol+'__')};base=sum(v[k] for k in v if k not in ['failed','done','length','blocked','off_axis_squared','overspeed_squared','failure_penalty']);required=[];episodes=[]
  for env in range(32):
   start=0
   for t in np.flatnonzero(v['done'][:,env]):
    b=base[start:t+1,env];L=len(b);isfail=bool(v['failed'][t,env]);G=float(np.dot(gamma**np.arange(L),b));den=1-gamma**(L-1) if isfail else 1
    req=max(0,-G/den) if den>1e-8 else None
    if req is not None:required.append(req)
    episodes.append({'length':L,'failed':isfail,'base_return':G,'required_drop_cost':req});start=t+1
  cuts={str(D):sum(e['base_return']-(gamma**(e['length']-1)*D if e['failed'] else 0)<-D for e in episodes)/max(1,len(episodes)) for D in [5,25,50,100,150,200,250,300]}
  rr={'completed_episodes':len(episodes),'terms':{k:stats(-v[k] if 'penalty' in k else v[k]) for k in ['off_axis_penalty','pose_penalty','overspeed_penalty','action_rate_penalty','limit_push_penalty','rotation','blocked']},'nonterminal_step_cost':stats(-base[~v['failed']]),'required_drop_cost':stats(required) if required else None,'fraction_observed_episodes_worse_than_immediate_minus_D':cuts}
  scene_report[protocol]=rr
  print(scene,protocol,'terms', {k:round(x['mean'],4) for k,x in rr['terms'].items()},'cost',rr['nonterminal_step_cost'],'required_D',rr['required_drop_cost'],'fraction',cuts)
 result[scene]=scene_report
(out/'calibration_summary.json').write_text(json.dumps(result,indent=2))
