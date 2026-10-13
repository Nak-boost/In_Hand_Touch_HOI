from pathlib import Path
import json,csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path('/root/In_Hand_Reconstruction')
OUT=Path(__file__).resolve().parent
objects=['sphere55','box50','capsule70'];colors=['#2463a6','#d27b23','#23836d']
tags=['episode_rewards/step','episode_lengths/step','rotation/speed','contact/gel_count','contact/housing_count','losses/bounds_loss']
labels=['Episode return','Episode duration (s)','Angular speed along Z (rad/s)','Active gel bodies (0–4)','Contacting housings (0–4)','Logged action penalty']
fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
summary={}
for obj,color in zip(objects,colors):
 run=ROOT/'runs/manip'/('teacher_'+obj+'_z_self_collision_v1')
 ea=EventAccumulator(str(run/'stage1_tb'),size_guidance={'scalars':0});ea.Reload()
 objsum={}
 with (OUT/(obj+'_scalars.csv')).open('w') as f:
  w=csv.writer(f);w.writerow(['tag','environment_steps','value'])
  for tag in ea.Tags()['scalars']:
   events=ea.Scalars(tag);values=np.array([e.value for e in events])
   assert np.isfinite(values).all(),(obj,tag)
   objsum[tag]=dict(first=float(values[0]),last=float(values[-1]),tail100_mean=float(values[-100:].mean()),tail1000_mean=float(values[-1000:].mean()))
   w.writerows((tag,e.step,e.value) for e in events)
   if tag in tags:
    ax=axs.flat[tags.index(tag)];x=np.array([e.step for e in events])/1e6
    if tag=='episode_lengths/step':values=values*.05
    smoothed=np.convolve(values,np.ones(100)/100,mode='valid')
    ax.plot(x[99:],smoothed,label=obj,color=color,lw=1.6)
 for which in ['best','last']:
  ck=torch.load(run/'stage1_nn'/(which+'.pth'),map_location='cpu')
  objsum[which]={k:ck.get(k) for k in ['agent_steps','epoch_num','best_rewards','last_lr']}
 ev=ea.Scalars('episode_rewards/step');objsum['wall_seconds']=ev[-1].wall_time-ev[0].wall_time
 summary[obj]=objsum
for ax,label in zip(axs.flat,labels):
 ax.set(xlabel='Environment steps (million)',ylabel=label);ax.grid(alpha=.2)
axs[0,1].axhline(30,color='gray',ls='--',lw=1)
axs[0,2].axhline(.5,color='gray',ls='--',lw=1,label='target')
axs[0,0].legend(frameon=False)
fig.suptitle('Privileged teacher training — rolling mean over 100 PPO updates')
fig.savefig(OUT/'training_curves.png',dpi=170)
fig.savefig(OUT/'training_curves.pdf')
(OUT/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({o:{k:summary[o][k] for k in ['best','last','wall_seconds']} for o in objects},indent=2))
