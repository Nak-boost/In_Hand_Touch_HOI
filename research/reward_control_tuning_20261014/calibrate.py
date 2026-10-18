"""Scripted physics probes and paired reward counterfactuals; no policy training."""
import isaacgym
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np,torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
p=argparse.ArgumentParser();p.add_argument('--scene',choices=['inhand','official','tabletop'],required=True);p.add_argument('--tag',default='');p.add_argument('--num-envs',type=int,default=32);p.add_argument('--steps',type=int,default=600);args=p.parse_args()
out=Path(__file__).parent;torch.set_num_threads(1)
with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
 cfg=compose(config_name={'inhand':'manip_compact_sphere55','official':'manip_official_sphere55','tabletop':'manip_tabletop_box45'}[args.scene],overrides=['stage=smoke','num_envs='+str(args.num_envs)])
np.random.seed(20261014);torch.manual_seed(20261014)
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='smoke'
env=(AllegroDigitTabletop if args.scene=='tabletop' else AllegroDigitTwoAxis)(task,'cuda:0',-1,True)
try:
 if args.scene=='inhand':
  with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
   donor=compose(config_name='manip_official_sphere55')
  cache=ROOT/donor.task.env.grasp.cache
 else:cache=ROOT/cfg.task.env.grasp.cache
 record=torch.load(cache,map_location='cpu')
 assert record['joint_names']==env.joint_names and record['objects']==env.catalog
 if args.scene!='inhand':assert record['asset_sha256']==env.asset_hash and record['collision_sha256']==env.collision_hash
 env.grasp_cache=record
 data={};start=time.perf_counter()
 for protocol in ['hold','small_noise','outward_push','alternating']:
  np.random.seed(42);torch.manual_seed(42);env.reset();saved={}
  for t in range(args.steps):
   if protocol=='hold':action=torch.zeros_like(env.actions)
   elif protocol=='small_noise':action=torch.rand_like(env.actions)*.4-.2
   elif protocol=='outward_push':action=torch.ones_like(env.actions)
   else:action=torch.ones_like(env.actions)*(1 if (t//100)%2==0 else -1)
   obs,reward,done,info=env.step(action)
   values=dict(env.reward_terms);values.update(env.reward_diagnostics)
   values.update(failed=env.terminal['failed'],done=env.terminal['done'],length=env.terminal['length'],blocked=env.blocked_target_command)
   assert torch.isfinite(reward).all()
   assert torch.allclose(sum(env.reward_terms.values()),reward,atol=1e-5)
   assert torch.equal(env.reward_terms['failure_penalty'], -env.cfg['drop_penalty']*env.terminal['failed'])
   assert (env.reward_terms['limit_push_penalty']<=0).all()
   if protocol=='hold':assert not env.blocked_target_command.any()
   for key,val in values.items():saved.setdefault(key,[]).append(val.detach().cpu().numpy().copy())
  data[protocol]={k:np.stack(v) for k,v in saved.items()}
  print(args.scene,protocol,'failed frames',int(data[protocol]['failed'].sum()),'seconds',round(time.perf_counter()-start,1),flush=True)
 flat={protocol+'__'+k:v for protocol,values in data.items() for k,v in values.items()};np.savez_compressed(out/(args.scene+args.tag+'_samples.npz'),**flat)
 meta={'scene':args.scene,'asset_sha256':env.asset_hash,'cache':str(cache),'cache_asset_sha256':record['asset_sha256'],'cache_matches_asset':record['asset_sha256']==env.asset_hash,'note':'Inhand uses historical states as diagnostic fixtures only, not approved training grasps. Official and tabletop use matching caches.','n_envs':args.num_envs,'steps_per_protocol':args.steps,'control_dt':env.control_dt,'gamma':cfg.train.ppo.gamma,'config':OmegaConf.to_container(cfg,resolve=True),'elapsed_seconds':time.perf_counter()-start}
 (out/(args.scene+args.tag+'_metadata.json')).write_text(json.dumps(meta,indent=2))
finally:env.close()
