from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;s=json.loads((p/'summary.json').read_text());data=json.loads((p/'scalars.json').read_text())
fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for ax,(tag,label,factor) in zip(axs.flat,[('rotation/speed','Signed Z speed (rad/s)',1),('episode_lengths/step','Episode duration (s)',.05),('episode_rewards/step','Episode return',1),('control/target_increment_lost','Target increment clipped (%)',100),('contact/gel_count','Gel force-threshold count (0-4)',1),('rotation/off_axis','Off-axis squared speed (rad²/s²)',1)]):
 for r in s['jobs']:
  a=np.asarray(data[r['object']][tag]);cut=(r['first_allocation_failure'] or {}).get('approx_after_steps',r['logged_steps']);a=a[a[:,0]<=cut];n=min(50,len(a));x=a[n-1:,0]/1e6;y=np.convolve(a[:,1]*factor,np.ones(n)/n,'valid');ax.plot(x,y,label=r['object'],linewidth=1.6)
 ax.set(xlabel='Environment steps (million)',ylabel=label);ax.grid(alpha=.22)
 if tag=='rotation/speed':ax.axhline(.5,color='gray',ls='--')
axs[0,0].legend(ncol=2,fontsize=9)
fig.suptitle('85-shape teacher experiments: final logs, 2026-10-14 19:25 CST\n50-update rolling means; truncated at first PhysX allocation failure',fontsize=13)
fig.savefig(p/'final_curves.png',dpi=150)
