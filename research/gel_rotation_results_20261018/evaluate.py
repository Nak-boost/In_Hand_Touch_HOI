"""Frozen-checkpoint first-episode diagnostic; never alters training processes."""
import argparse,json,sys,math,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--group',choices=['single','multi'],required=True);p.add_argument('--checkpoint',choices=['last','best'],required=True);p.add_argument('--seed',type=int,default=215);p.add_argument('--per-object',type=int,default=64)
a=p.parse_args();OUT=Path(__file__).resolve().parent
meta=json.loads((OUT/'training_summary.json').read_text())[a.group];run=Path(meta['run']);ckpath=Path(meta['checkpoints'][a.checkpoint]['path'])
sys.path.insert(0,str(run/'source'))
import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
from omegaconf import OmegaConf
import hora.tasks.allegro_digit_two_axis as task_module
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');task_module.ROOT=ROOT
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import require_action_reference
from scripts.check_grasp_cache import install,contact_pairs
cfg=OmegaConf.load(run/'config.yaml');catalog=OmegaConf.to_container(cfg.task.env.objects,resolve=True)
n=a.per_object*len(catalog);cfg.task.env.numEnvs=n;cfg.task.rl_device='cpu';cfg.task.sim.use_gpu_pipeline=False
np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(1)
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
env=task_module.AllegroDigitTwoAxis(task,'cuda:0',-1,True)
try:
 ck=torch.load(ckpath,map_location='cpu');require_collision_signature(ck,env.collision_hash,'Teacher');require_action_reference(ck,env.cfg);assert ck['asset_sha256']==env.asset_hash
 model=ActorCritic(network_spec(cfg,env.num_obs)).eval();model.load_state_dict(ck['model'])
 rms=RunningMeanStd((env.num_obs,)).eval();rms.load_state_dict(ck['running_mean_std'])
 bank=torch.load(ROOT/cfg.task.env.grasp.cache,map_location='cpu');states=torch.empty(n,23);targets=torch.empty(n,16);chosen={}
 rng=np.random.RandomState(a.seed)
 for oid,obj in enumerate(catalog):
  name=obj['name'];idx=rng.choice(len(bank['states'][name]),a.per_object,replace=False);chosen[name]=idx.tolist();mask=env.object_ids==oid
  states[mask]=bank['states'][name][idx];targets[mask]=bank['targets'][name][idx]
 env.stage='grasps';install(env,states,targets);obs=env._observations()
 alive=torch.ones(n,dtype=torch.bool);failed=torch.zeros(n,dtype=torch.bool);turns=torch.zeros(n);duration=torch.zeros(n);returns=torch.zeros(n);counts=torch.zeros(n);lateangle=torch.zeros(n);latecount=torch.zeros(n);ever=torch.zeros(n,dtype=torch.bool)
 sums={};noncount=torch.zeros(n);paircount=torch.zeros(n);trace=[];bodycounts={};body_names=env.gym.get_asset_rigid_body_names(env.hand_asset)
 for step in range(600):
  with torch.no_grad():
   mu,*_=model._actor_critic({'obs':rms(obs['obs']),'priv_info':obs['priv_info']})
   obs,rew,_,_=env.step(torch.where(alive[:,None],mu,torch.zeros_like(mu)).clamp(-1,1))
  assert torch.isfinite(rew).all() and torch.isfinite(env.object_state).all()
  newturns=env.total_angle[:,0]/(2*math.pi);dtheta=(newturns-turns)*2*math.pi
  g=env.task_gel_contacts.sum(-1);displacement=(env.object_state[:,:3]-env.initial_position).norm(dim=-1)
  terms={'speed':dtheta/env.control_dt,'two_gels':(g>=2).float(),'no_gel':(g==0).float(),'gel_count':g.float(),'displacement':displacement,'raw_action_clipped':(mu.abs()>1).float().mean(-1),'target_lost':env.target_increment_lost,'target_at_limit':((env.targets-env.lower<.001)|(env.upper-env.targets<.001)).float().mean(-1),'off_axis_squared':env.reward_diagnostics['off_axis_squared'],**{'reward_'+k:v for k,v in env.reward_terms.items()}}
  for k,v in terms.items():
   if k not in sums:sums[k]=torch.zeros(n)
   sums[k][alive]+=v[alive]
  counts+=alive.float();turns[alive]=newturns[alive];duration[alive]=(step+1)*env.control_dt;returns[alive]+=rew[alive];ever|=alive&(turns>=1)
  if step>=500:lateangle[alive]+=dtheta[alive];latecount+=alive.float()
  if step%10==0:
   _,non=contact_pairs(env,alive);noncount[alive]+=non[alive].float();paircount+=alive.float()
   for i in alive.nonzero().flatten().tolist():
    ob=env.gym.get_actor_rigid_body_index(env.envs[i],1,0,gymapi.DOMAIN_ENV);touched=set()
    for c in env.gym.get_env_rigid_contacts(env.envs[i]):
     x,y=int(c['body0']),int(c['body1']);other=y if x==ob else x if y==ob else -1
     if c['lambda']>0 and 0<=other<len(body_names):touched.add(body_names[other])
    for key in touched:
     if key not in bodycounts:bodycounts[key]=torch.zeros(n)
     bodycounts[key][i]+=1
  failed|=alive&env.terminal['failed'];alive &= ~env.terminal['done']
  if (step+1)%100==0:
   trace.append({'seconds':(step+1)*env.control_dt,'per_object':{obj['name']:{'mean_turns':float(turns[env.object_ids==oid].mean()),'retained':int((~failed&(env.object_ids==oid)).sum())} for oid,obj in enumerate(catalog)}})
   print(a.group,a.checkpoint,'step',step+1,'alive',int(alive.sum()),flush=True)
  if not alive.any():break
 rows={}
 for oid,obj in enumerate(catalog):
  mask=env.object_ids==oid;survived=mask&(~failed)&(duration>=29.99);lmask=survived&(latecount>0);ls=lateangle/(latecount*env.control_dt).clamp_min(1e-6)
  rows[obj['name']]={'n':int(mask.sum()),'retained_30s':float(survived[mask].float().mean()),'one_turn_retained':float((survived&(turns>=1))[mask].float().mean()),'ever_one_turn':float(ever[mask].float().mean()),'mean_turns':float(turns[mask].mean()),'median_turns':float(turns[mask].median()),'mean_seconds':float(duration[mask].mean()),'mean_return':float(returns[mask].mean()),'last5s_speed_survivors':float(ls[lmask].mean()) if lmask.any() else None,'last5s_abs_speed_lt_005_survivors':float((ls[lmask].abs()<.05).float().mean()) if lmask.any() else None,'frame_means':{k:float(v[mask].sum()/counts[mask].sum().clamp_min(1)) for k,v in sums.items()},'nongel_object_frames':float(noncount[mask].sum()/paircount[mask].sum().clamp_min(1)),'body_contact_frame_fraction':{k:float(v[mask].sum()/paircount[mask].sum().clamp_min(1)) for k,v in bodycounts.items()}}
 result={'group':a.group,'checkpoint':a.checkpoint,'checkpoint_steps':ck['agent_steps'],'checkpoint_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),'seed':a.seed,'per_object':rows,'trace':trace,'chosen_cache_indices':chosen,'protocol':'Deterministic privileged teacher; first episode max30s; original model/control/domain/reward; frames after termination excluded; GPU PhysX CPU inference; nongel contacts sampled every10steps.','episodes':{'object_ids':env.object_ids.tolist(),'failed':failed.tolist(),'turns':turns.tolist(),'seconds':duration.tolist(),'late_angle':lateangle.tolist(),'late_count':latecount.tolist()}}
 path=OUT/f'eval_{a.group}_{a.checkpoint}_{a.seed}.json';path.write_text(json.dumps(result,indent=2));print('RESULT_SAVED',path,flush=True)
finally:
 print('CLEANUP_START',flush=True);env.close();print('CLEANUP_DONE',flush=True)
