import isaacgym
from isaacgym import gymapi
import json,argparse
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--object',required=True);args=p.parse_args()
obj=args.object;run=ROOT.parent/'runs/manip'/('teacher_'+obj+'_z_self_collision_v1')
cfg=OmegaConf.load(run/'config.yaml');cfg.task.env.numEnvs=64;cfg.task.sim.use_gpu_pipeline=False;cfg.task.rl_device='cpu'
np.random.seed(20261013);torch.manual_seed(20261013)
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
env=AllegroDigitTwoAxis(task,'cuda:0',-1,True)
try:
 ck=torch.load(run/'stage1_nn/last.pth',map_location='cpu')
 require_collision_signature(ck,env.collision_hash,'Teacher')
 model=ActorCritic(network_spec(cfg,env.num_obs)).eval();model.load_state_dict(ck['model'])
 rms=RunningMeanStd((env.num_obs,)).eval();rms.load_state_dict(ck['running_mean_std'])
 torch.manual_seed(20261014);obs=env.reset();env.stage='grasps'
 alive=torch.ones(64,dtype=torch.bool);count=0
 names=env.gym.get_asset_rigid_body_names(env.hand_asset)
 object_id=env.gym.get_actor_rigid_body_index(env.envs[0],1,0,gymapi.DOMAIN_ENV)
 sums={k:0 for k in ['object_palm','object_housing','object_gel','object_gel_ge2','object_non_gel','hand_self_contact']}
 pairs={};trace=[]
 for step in range(600):
  with torch.no_grad():
   raw=model.act_inference({'obs':rms(obs['obs']),'priv_info':obs['priv_info']}).clamp(-1,1)
   raw[~alive]=0
   obs,_,_,_=env.step(raw)
  count+=int(alive.sum())
  for i in alive.nonzero(as_tuple=False).flatten().tolist():
   touching=set();self_contact=False
   for c in env.gym.get_env_rigid_contacts(env.envs[i]):
    a,b=int(c['body0']),int(c['body1'])
    if c['lambda']<=0:continue
    if 0<=a<len(names) and 0<=b<len(names):self_contact=True
    other=b if a==object_id else a if b==object_id else -1
    if 0<=other<len(names):touching.add(other)
   for idx in touching:pairs[names[idx]]=pairs.get(names[idx],0)+1
   gels=len(touching.intersection(env.gel_ids))
   sums['object_palm']+=int(names.index('base_link') in touching)
   sums['object_housing']+=int(bool(touching.intersection(env.housing_ids)))
   sums['object_gel']+=gels
   sums['object_gel_ge2']+=int(gels>=2)
   sums['object_non_gel']+=int(bool(touching.difference(env.gel_ids)))
   sums['hand_self_contact']+=int(self_contact)
  alive &= ~env.terminal['done']
  if step in [19,99,299,599]:
   trace.append(dict(seconds=(step+1)*env.control_dt,remaining=int(alive.sum()),mean_object_z=float(env.object_state[alive,2].mean()) if alive.any() else None))
  if not alive.any():break
 result=dict(object=obj,checkpoint='last',n=64,method='GPU PhysX, CPU contact API, first episode only, contact lambda>0',active_frames=count,frame_fractions={k:v/count for k,v in sums.items()},object_contact_body_fraction={k:v/count for k,v in pairs.items()},trace=trace)
 (OUT/(obj+'_contact_pairs.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
finally:env.close()
