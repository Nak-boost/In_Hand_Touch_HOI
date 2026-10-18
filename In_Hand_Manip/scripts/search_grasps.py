"""Search real-contact grasps, refining successful physical states without changing physics."""
import isaacgym
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from check_grasp_cache import install,contact_pairs

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output-cache',required=True)
p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=128);p.add_argument('--envs',type=int,default=256);p.add_argument('--batches',type=int,default=40);p.add_argument('--seed-cache');p.add_argument('--extra-preload',type=float,default=0.)
a=p.parse_args();dest=ROOT/a.output_cache
if dest.exists():raise FileExistsError(dest)
np.random.seed(42);torch.manual_seed(42)
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
 cfg=compose(config_name=a.config,overrides=['stage=grasps','pipeline=cpu','rl_device=cpu',f'num_envs={a.envs}','graphics_device_id=-1'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='grasps';env=AllegroDigitTwoAxis(task,'cuda:0',-1,True)
name=env.catalog[0]['name'];rows=[];elite=torch.empty((0,23));elite_targets=torch.empty((0,16));report=[]
if a.seed_cache:
 previous=torch.load(ROOT/a.seed_cache,map_location='cpu')
 assert previous['asset_sha256']==env.asset_hash and previous['collision_sha256']==env.collision_hash
 assert previous['objects']==env.catalog and previous['joint_names']==env.joint_names
 elite=previous['states'][name].clone();elite_targets=previous.get('targets',{}).get(name,elite[:,:16]).clone()
starting_count=len(elite)
try:
 for batch in range(a.batches):
  states=torch.zeros(a.envs,23);states[:,:16]=torch.tensor(cfg.task.env.grasp.canonical_q)[env.canonical_order]
  states[:,:16]+=(torch.rand(a.envs,16)*2-1)*float(cfg.task.env.grasp.q_noise)
  states[:,16:19]=torch.tensor(cfg.task.env.grasp.object_position);states[:,16:18]+=(torch.rand(a.envs,2)*2-1)*.002;states[:,22]=1
  preload=torch.tensor(cfg.task.env.grasp.get('preload_delta',[0.]*16))[env.canonical_order]
  targets=states[:,:16]+preload*(.5+torch.rand(a.envs,1))
  if len(elite):
   k=3*a.envs//4;sample_ids=torch.randint(len(elite),(k,));states[:k]=elite[sample_ids];noise=(torch.rand(k,16)*2-1)*.008;states[:k,:16]+=noise;targets[:k]=elite_targets[sample_ids]+noise
   states[:k,16:18]+=(torch.rand(k,2)*2-1)*.0005
  states[:,:16]=torch.maximum(torch.minimum(states[:,:16],env.upper),env.lower)
  targets+=preload*a.extra_preload*torch.rand(a.envs,1)
  targets=torch.maximum(torch.minimum(targets,env.upper),env.lower)
  install(env,states,targets);valid=torch.ones(a.envs,dtype=torch.bool);gcounts=torch.zeros(a.envs);nong=torch.zeros(a.envs)
  for step in range(80):
   env.step(torch.zeros_like(env.actions))
   if step>=25:
    valid &= (~env.terminal['failed']) & (env.object_state[:,7:10].norm(dim=-1)<.1) & (env.object_state[:,10:13].norm(dim=-1)<1.)
    gel,non=contact_pairs(env,valid)
    gcounts+=(gel.sum(-1)>=2).float();nong+=non.float()
  physical=torch.cat([env.q,env.object_state[:,:7]],-1).clone()
  basic=valid&(gcounts>=53)&(nong<=5)
  new=physical[basic];elite=torch.cat([elite,new]);elite_targets=torch.cat([elite_targets,env.targets[basic].clone()])
  record=dict(batch=batch+1,accepted=len(new),total=len(elite),retained=int(valid.sum()),two_gel=int((valid&(gcounts>=53)).sum()),non_gel_free=int((valid&(nong<=5)).sum()))
  report.append(record);print(json.dumps(record),flush=True)
  if len(elite)-starting_count>=a.count:break
 if len(elite)==starting_count:raise RuntimeError('No usable initial grasps; inspect geometry and placement.')
 bank=dict(asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,grasp_contact_validation='physx_body_pairs_refined_v1',objects=env.catalog,joint_names=env.joint_names,states={name:elite[starting_count:starting_count+a.count]},targets={name:elite_targets[starting_count:starting_count+a.count]},generation=dict(seed=42,seed_cache=a.seed_cache,extra_preload=a.extra_preload,config=OmegaConf.to_container(cfg,resolve=True),batches=report))
 dest.parent.mkdir(parents=True,exist_ok=True);torch.save(bank,dest)
 Path(a.output).write_text(json.dumps(dict(config=a.config,count=min(len(elite)-starting_count,a.count),batches=report),indent=2)+'\n')
 print('Saved candidates:',dest,min(len(elite)-starting_count,a.count),flush=True)
finally:env.close()
