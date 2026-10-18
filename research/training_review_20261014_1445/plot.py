from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
s=json.loads((p/'summary.json').read_text()); series=json.loads((p/'scalars.json').read_text())
for r in s['jobs']:
 limit=(r.get('first_allocation_failure') or {}).get('approx_after_steps',r.get('steps',0))
 r['pre_error_last100']={}
 for tag,(steps,values,times) in series[r['object']].items():
  v=[v for t,v in zip(steps,values) if t<=limit][-100:]
  r['pre_error_last100'][tag]=float(np.mean(v))
(p/'summary.json').write_text(json.dumps(s,indent=2))
fig, axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
tags=[('rotation/speed','Signed Z speed (rad/s)',1),('episode_lengths/step','Episode duration (s)',.05),('episode_rewards/step','Episode return',1),('control/target_increment_lost','PD target increment clipped (%)',100),('contact/gel_count','Gel force-threshold count (0-4)',1),('rotation/off_axis','Off-axis squared speed (rad²/s²)',1)]
for ax,(tag,label,factor) in zip(axes.flat,tags):
 for row in s['jobs']:
  name=row['object']; x,y,_=series[name][tag];x=np.array(x);y=np.array(y)*factor
  lim=(row.get('first_allocation_failure') or {}).get('approx_after_steps',row['steps']); keep=x<=lim;x=x[keep];y=y[keep]
  n=min(50,len(y)); yy=np.convolve(y,np.ones(n)/n,mode='valid');xx=x[n-1:]/1e6
  ax.plot(xx,yy,label=name,linewidth=1.7)
 ax.set(xlabel='Environment steps (million)',ylabel=label);ax.grid(alpha=.22)
 if tag=='rotation/speed':ax.axhline(.5,color='gray',linestyle='--',alpha=.7)
axes[0,0].legend(ncol=2,fontsize=9)
fig.suptitle('Teacher PPO: 50-update rolling means; curves truncated at first allocation failure\nSnapshot 2026-10-14 14:46 CST; capsule80 has only 14 updates',fontsize=13)
fig.savefig(p/'training_curves.png',dpi=160)
for r in s['jobs']:
 m=r['pre_error_last100'];print(r['object'], {k:round(m[k],4) for k in ['rotation/speed','rotation/off_axis','episode_lengths/step','episode_rewards/step','reward/pose_penalty','control/target_increment_lost','control/target_at_limit','control/torque_at_limit','contact/gel_count','contact/housing_count']})
