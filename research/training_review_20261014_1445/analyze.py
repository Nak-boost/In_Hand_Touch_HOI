from pathlib import Path
from datetime import datetime
import json, math
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out=Path(__file__).parent
root=Path('/root/In_Hand_Reconstruction')
queue=json.loads((root/'research/grasp_training_20261014/queue_status.json').read_text())
result={'captured_at':datetime.now().astimezone().isoformat(),'window_updates':100,'jobs':[]}
series={}
for job in queue['jobs']:
 p=root/'runs/manip'/job['run_name']; row={'object':job['object'],'status':job['status'],'run':str(p),'metrics':{},'checkpoints':{}}; series[job['object']]={}
 for event in p.rglob('events.out.tfevents.*'):
  a=EventAccumulator(str(event),size_guidance={'scalars':0}); a.Reload()
  for tag in a.Tags()['scalars']:
   events=a.Scalars(tag); values=np.array([e.value for e in events]); steps=np.array([e.step for e in events]); times=np.array([e.wall_time for e in events])
   series[job['object']][tag]=[steps.tolist(),values.tolist(),times.tolist()]
   row['metrics'][tag]={'n':len(values),'last_step':int(steps[-1]),'first100_mean':float(values[:100].mean()),'last100_mean':float(values[-100:].mean()),'previous100_mean':float(values[-200:-100].mean()) if len(values)>100 else None,'last':float(values[-1]),'max':float(values.max()),'nonfinite':int((~np.isfinite(values)).sum())}
   if tag=='rotation/speed':
    row['steps']=int(steps[-1]); row['last_logged_at']=datetime.fromtimestamp(times[-1]).astimezone().isoformat()
    row['recent_wall_fps']=float((steps[-1]-steps[-min(100,len(steps))])/(times[-1]-times[-min(100,len(steps))]))
    row['early_wall_fps']=float((steps[min(99,len(steps)-1)]-steps[0])/(times[min(99,len(steps)-1)]-times[0]))
 for name in ['best','last']:
  cp=p/'stage1_nn'/f'{name}.pth'
  if not cp.exists(): continue
  state=torch.load(cp,map_location='cpu')
  stats={k:state.get(k) for k in ['agent_steps','epoch_num','best_rewards','last_lr','asset_sha256','collision_sha256']}
  stats['all_model_weights_finite']=all(bool(torch.isfinite(v).all()) for v in state['model'].values())
  stats['sigma_parameter']=state['model']['sigma'].tolist()
  row['checkpoints'][name]=stats
 result['jobs'].append(row)
(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
(out/'scalars.json').write_text(json.dumps(series,allow_nan=False))
tags=['rotation/speed','rotation/off_axis','episode_rewards/step','episode_lengths/step','reward/pose_penalty','control/target_increment_lost','control/target_at_limit','control/torque_at_limit','contact/gel_count','contact/housing_count','episode/failed','episode/time_limit','losses/bounds_loss','losses/entropy','info/kl']
for r in result['jobs']:
 print('\n',r['object'],r['status'],r.get('steps'),r.get('last_logged_at'),'fps',r.get('early_wall_fps'),r.get('recent_wall_fps'))
 for t in tags:
  if t in r['metrics']:
   x=r['metrics'][t];print(t, 'early %.5f recent %.5f prev %s'%(x['first100_mean'],x['last100_mean'],str(x['previous100_mean'])))
 print('checkpoints',r['checkpoints'])
