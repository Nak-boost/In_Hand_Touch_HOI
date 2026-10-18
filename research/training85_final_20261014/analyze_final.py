from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,shutil,hashlib
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out=Path(__file__).parent;root=Path('/root/In_Hand_Reconstruction')
queue=json.loads((out/'queue_status_after_stop.json').read_text());old=json.loads((root/'research/training_review_20261014_1445/summary.json').read_text())
old={r['object']:r for r in old['jobs']};series={};result={'captured_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'window_updates':100,'jobs':[]}
for job in queue['jobs']:
 name=job['object'];p=root/'runs/manip'/job['run_name'];snap=out/'snapshots'/name;snap.mkdir(parents=True,exist_ok=True)
 for f in ['config.yaml','asset_metadata.json','allegro_digit.urdf']:
  shutil.copy2(p/f,snap/f)
 log=Path(job['console_log']);shutil.copy2(log,snap/'console.log');n=0;fail=None;errors=[]
 for number,line in enumerate(log.read_text(errors='replace').splitlines(),1):
  if line.startswith('Agent Steps:'):n+=1
  if 'fail to allocate memory' in line and fail is None:fail={'line':number,'after_logged_updates':n,'approx_after_steps':n*4096,'message':line}
  if any(s in line for s in ['fail to allocate memory','Simulation can not continue','RuntimeError:','Segmentation','KeyboardInterrupt']):errors.append({'line':number,'message':line})
 row={'object':name,'status':job['status'],'run':str(p),'first_allocation_failure':fail,'errors':errors,'metrics':{},'checkpoints':{}}
 data={}
 for event in p.rglob('events.out.tfevents.*'):
  target=snap/event.name;shutil.copy2(event,target);a=EventAccumulator(str(target),size_guidance={'scalars':0});a.Reload()
  for tag in a.Tags()['scalars']:data.setdefault(tag,[]).extend([[e.step,e.value,e.wall_time] for e in a.Scalars(tag)])
 series[name]=data
 for tag,events in data.items():
  events.sort(key=lambda x:x[0]);steps=np.array([e[0] for e in events]);values=np.array([e[1] for e in events]);times=np.array([e[2] for e in events]);valid=values[steps<=fail['approx_after_steps']] if fail else values
  row['metrics'][tag]={'first100':float(values[:100].mean()),'last100':float(values[-100:].mean()),'previous100':float(values[-200:-100].mean()),'pre_error_last100':float(valid[-100:].mean()),'nonfinite_count':int((~np.isfinite(values)).sum())}
  if tag=='rotation/speed':
   row.update(logged_steps=int(steps[-1]),logged_updates=len(events),last_logged_at=datetime.fromtimestamp(times[-1],timezone(timedelta(hours=8))).isoformat(),recent_fps=float((steps[-1]-steps[-100])/(times[-1]-times[-100])))
 for name_cp in ['best','last']:
  f=p/'stage1_nn'/f'{name_cp}.pth'
  if not f.exists():continue
  target=snap/f.name;shutil.copy2(f,target);state=torch.load(target,map_location='cpu')
  row['checkpoints'][name_cp]={k:state.get(k) for k in ['agent_steps','epoch_num','best_rewards','last_lr','asset_sha256','collision_sha256']}
  row['checkpoints'][name_cp].update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),model_finite=all(bool(torch.isfinite(v).all()) for v in state['model'].values()))
 row['previous_review_steps']=old[name].get('steps');row['previous_review_metrics']={k:v['last100_mean'] for k,v in old[name]['metrics'].items()}
 result['jobs'].append(row)
(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));(out/'scalars.json').write_text(json.dumps(series,allow_nan=False))
for r in result['jobs']:
 print('\n',r['object'],r['status'],r['logged_steps'],r['last_logged_at'],'recentfps',round(r['recent_fps'],1),'first allocfail',r['first_allocation_failure'])
 for tag in ['rotation/speed','rotation/off_axis','episode_rewards/step','episode_lengths/step','control/target_increment_lost','control/target_at_limit','control/torque_at_limit','contact/gel_count','contact/housing_count','reward/pose_penalty','losses/bounds_loss']:
  m=r['metrics'][tag];print(tag, 'first %.4f last %.4f pre_error %.4f prev_window %.4f'%(m['first100'],m['last100'],m['pre_error_last100'],m['previous100']))
 print('checkpoint steps', {k:v['agent_steps'] for k,v in r['checkpoints'].items()})
 print('errors',r['errors'][:3])
