import faulthandler
faulthandler.enable()
import isaacgym
from isaacgym import gymapi
import argparse,json,os
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.manip_control import tracking_action
from scripts.check_grasp_cache import install
p=argparse.ArgumentParser();p.add_argument('--pipeline',choices=['cpu','gpu'],required=True);a=p.parse_args()
np.random.seed(215);torch.manual_seed(215);torch.set_num_threads(1)
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
 cfg=compose(config_name='manip_tabletop_box45',overrides=['num_envs=4','pipeline='+a.pipeline,'rl_device=cpu','graphics_device_id=-1','num_subscenes=1'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
env=AllegroDigitTabletop(task,'cuda:0',-1,True);env.stage='grasps'
bank=torch.load(env.cache_path,map_location=env.device);name=env.catalog[0]['name'];states=bank['states'][name][:4];targets=bank['targets'][name][:4]
print('before_reset',flush=True)
env.reset()
print('after_reset',flush=True)
targets=env.targets.clone()
print('installed',flush=True)
for step in range(40):
 env.step(tracking_action(env.q,env.targets,targets,env.actions,env.cfg))
 if step in (0,39):print('stepped',step,flush=True)
if a.pipeline=='gpu':torch.cuda.synchronize()
rows=[]
for i,e in enumerate(env.envs):
 ob=env.gym.get_actor_rigid_body_index(e,1,0,gymapi.DOMAIN_ENV)
 print('before_env_contacts',i,flush=True)
 cs=env.gym.get_env_rigid_contacts(e)
 print('dtype',getattr(cs,'dtype',None),flush=True)
 rows.append(dict(env=i,object=ob,n=len(cs),netgel=env.contact_forces[i,env.gel_ids].cpu().tolist(),gel_pairs=[dict(body0=int(c['body0']),body1=int(c['body1']),lam=float(c['lambda']),normal=[float(v) for v in c['normal']]) for c in cs if int(c['body0']) in env.gel_ids or int(c['body1']) in env.gel_ids]))
r=dict(pipeline=a.pipeline,device=env.device,gel_ids=env.gel_ids,dt=env.dt,rows=rows)
from pathlib import Path
(Path(__file__).resolve().parent/('probe_'+a.pipeline+'.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
os._exit(0)
