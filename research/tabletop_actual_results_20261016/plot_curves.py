import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for seed in (42,43,44):
 d=json.loads((p/f'scalars_{seed}.json').read_text())
 for ax,tag,title,factor in zip(axes.flat,['episode_lengths/step','rotation/speed','control/target_increment_lost','losses/bounds_loss'],['Episode duration (s)','Signed Z speed (rad/s; sampled log)','Requested increment lost (%)','Action mean bounds loss (unweighted)'],[.05,1,100,1]):
  xs=d[tag];x=np.array([i['step']/1e6 for i in xs]);y=np.array([i['value']*factor for i in xs]);w=50
  ax.plot(x[w-1:],np.convolve(y,np.ones(w)/w,mode='valid'),label=f'seed {seed}',lw=1.3)
  ax.set(title=title,xlabel='Environment steps (million)');ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Tabletop box45: actual-position control + fingertip proximity reward\n50-update moving averages; rotation log samples last rollout step')
fig.savefig(p/'training_curves.png',dpi=160)
