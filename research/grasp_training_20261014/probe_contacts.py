import isaacgym
from isaacgym import gymapi
import json, argparse
import torch
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
p=argparse.ArgumentParser();p.add_argument('--config',default='manip_official_sphere55');a=p.parse_args()
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
 cfg=compose(config_name=a.config,overrides=['stage=smoke','pipeline=cpu','rl_device=cpu','num_envs=16','graphics_device_id=-1'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='grasps';env=AllegroDigitTwoAxis(task,'cuda:0',-1,True)
names=env.gym.get_asset_rigid_body_names(env.hand_asset)+['object']
try:
 env.reset()
 for step in range(80):
  env.step(torch.zeros_like(env.actions))
  if step in [0,4,19,39,79]:
   rows=[]
   for i in range(4):
    pairs=[]
    for c in env.gym.get_env_rigid_contacts(env.envs[i]):
     if c['lambda']>0:pairs.append([names[int(c['body0'])] if c['body0']>=0 else 'ground',names[int(c['body1'])] if c['body1']>=0 else 'ground'])
    rows.append(dict(env=i,pos=env.object_state[i,:3].tolist(),gels=env.raw_contacts[i].tolist(),failed=bool(env.terminal['failed'][i]),pairs=pairs))
   print(json.dumps(dict(step=step,mean_z=float(env.object_state[:,2].mean()),qerr=float((env.targets-env.q).abs().mean()),rows=rows)),flush=True)
finally:env.close()
