from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
objects=['sphere55','sphere65','box45','box55','capsule70','capsule80']
colors=['#0072B2','#56B4E9','#D55E00','#E69F00','#009E73','#CC79A7']
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
metrics=[('rotation/speed','Z angular speed (rad/s)',1),('episode_lengths/step','Episode duration (s)',.05),
         ('control/target_increment_lost','Discarded target increment (%)',100),('episode_rewards/step','Episode return',1)]
for obj,c in zip(objects,colors):
    data=json.loads((OUT/f'scalars_{obj}.json').read_text())
    for ax,(tag,label,scale) in zip(axs.flat,metrics):
        xs=data[tag]
        for_draw=[xs[i:i+40] for i in range(0,len(xs),40)]
        ax.plot([np.mean([s['step'] for s in b])/1e6 for b in for_draw],
                [np.mean([s['value'] for s in b])*scale for b in for_draw],label=obj,color=c,lw=1.8)
        ax.set(xlabel='Environment steps (million)',ylabel=label);ax.grid(alpha=.2)
axs[0,0].axhline(.5,color='gray',ls='--',lw=1,label='Target speed')
axs[0,1].set_ylim(0,31);axs[1,0].set_ylim(0,100)
axs[0,0].legend(ncol=2,fontsize=8)
fig.savefig(OUT/'training_curves.png',dpi=170);plt.close(fig)
if all((OUT/f'eval_{o}_mean_last.json').exists() for o in objects):
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for obj,c in zip(objects,colors):
        r=json.loads((OUT/f'eval_{obj}_mean_last.json').read_text())
        ax.plot([0]+[x['seconds'] for x in r['trace']], [0]+[x['mean_turns'] for x in r['trace']],label=obj,color=c,marker='o')
    ax.axhline(1,color='gray',ls='--',lw=1)
    ax.set(xlabel='Replay time (s)',ylabel='Mean accumulated Z rotation (turns)')
    ax.grid(alpha=.2);ax.legend(ncol=3)
    fig.savefig(OUT/'replay_rotation.png',dpi=170)
