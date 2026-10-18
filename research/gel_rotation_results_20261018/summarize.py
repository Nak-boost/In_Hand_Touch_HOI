from pathlib import Path
import json,shutil,math,numpy as np,torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
OUT=Path(__file__).resolve().parent
runs={'single':Path('/root/In_Hand_Reconstruction/runs/manip/teacher_sphere55_gel16384_20261017_resumed_attempt00'),'multi':Path('/root/In_Hand_Reconstruction/runs/manip/teacher_six_objects_gel16384_direct_20261017_152146')}
summary={}
for name,run in runs.items():
 scalars={}
 for f in sorted(run.rglob('events.out.tfevents.*')):
  ea=EventAccumulator(str(f),size_guidance={'scalars':0});ea.Reload()
  for tag in ea.Tags()['scalars']:
   rows=ea.Scalars(tag);scalars[tag]=np.array([[x.step,x.value,x.wall_time] for x in rows])
 np.savez_compressed(OUT/(name+'_scalars.npz'),**scalars)
 end=max(v[-1,0] for v in scalars.values() if len(v));bad=[k for k,v in scalars.items() if not np.isfinite(v).all()]
 checkpoints={}
 for ck in ['last','best']:
  dest=OUT/(name+'_'+ck+'.pth');shutil.copyfile(run/'stage1_nn'/(ck+'.pth'),dest)
  data=torch.load(dest,map_location='cpu');checkpoints[ck]={'path':str(dest),'agent_steps':data['agent_steps'],'epoch_num':data['epoch_num']}
 windows={}
 for label,lo,hi in [('0-10M',0,10e6),('90-100M',90e6,100e6),('190-200M',190e6,200e6),('290-300M',290e6,300e6),('390-400M',390e6,400e6),('last10M',end-10e6,end),('previous10M',end-20e6,end-10e6)]:
  windows[label]={k:float(v[(v[:,0]>=lo)&(v[:,0]<=hi),1].mean()) for k,v in scalars.items() if ((v[:,0]>=lo)&(v[:,0]<=hi)).any()}
 summary[name]={'run':str(run),'latest_steps':int(end),'nonfinite_tags':bad,'checkpoints':checkpoints,'windows':windows,'tags':list(scalars)}
(OUT/'training_summary.json').write_text(json.dumps(summary,indent=2))
for name,s in summary.items():
 print(name,'steps',s['latest_steps'],'checkpoints',s['checkpoints'],'nonfinite',s['nonfinite_tags'])
 print('TAGS',s['tags'])
 for label in ['390-400M','last10M','previous10M']:
  print(label,{k:v for k,v in s['windows'][label].items() if any(t in k for t in ['episode','rotation/','contact/task','reward/','control/','limit','clip'])})
