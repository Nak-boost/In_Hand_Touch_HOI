from pathlib import Path
import json
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
OUT=Path(__file__).resolve().parent
runs=OUT.parent.parent/'runs/manip'
result={}
for seed in ("sphere55","box45","sphere65","capsule70","box55","capsule80"):
 r=runs/f'teacher_{seed}_z_compact27_reward250_resume20261016'
 ea=EventAccumulator(str(r/'stage1_tb'),size_guidance={'scalars':0});ea.Reload()
 tags=ea.Tags()['scalars']
 scalars={t:[{'step':x.step,'value':x.value} for x in ea.Scalars(t)] for t in tags}
 ckinfo={}
 for name in ('last','best'):
  ck=torch.load(r/f'stage1_nn/{name}.pth',map_location='cpu')
  ckinfo[name]={k:v for k,v in ck.items() if isinstance(v,(str,int,float,bool))}
 means={t:float(np.mean([x['value'] for x in xs[-100:]])) for t,xs in scalars.items()}
 windows={}
 for end in (1000000,3000000,5000000,8000000,10002432):
  windows[end]={t:float(np.mean([x['value'] for x in xs if end-1000000<x['step']<=end])) for t,xs in scalars.items() if any(end-1000000<x['step']<=end for x in xs)}
 result[seed]={'checkpoints':ckinfo,'last100':means,'windows':windows,'last_step':max(x['step'] for xs in scalars.values() for x in xs)}
 (OUT/f'scalars_{seed}.json').write_text(json.dumps(scalars))
 print(seed,json.dumps(result[seed],indent=2))
(OUT/'training_summary.json').write_text(json.dumps(result,indent=2))
