import isaacgym
from isaacgym import gymtorch
import argparse,json,time,os,subprocess,threading
from pathlib import Path
import numpy as np,torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.utils.mount_geometry import POSTURES
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--benchmark',action='store_true');p.add_argument('--sweep',action='store_true');p.add_argument('--rollout',action='store_true');p.add_argument('--steps',type=int,default=120);args=p.parse_args()
out=Path(__file__).parent;memory=[];stop=threading.Event()
def sample():
 while not stop.is_set():
  try:
   txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
   for line in txt.splitlines():
    a,b=line.split(',')
    if int(a)==os.getpid():memory.append({'time':time.time(),'mib':int(b)})
  except Exception:pass
  stop.wait(1)
thread=threading.Thread(target=sample,daemon=True);thread.start()
np.random.seed(14);torch.manual_seed(14)
with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
 cfg=compose(config_name='manip',overrides=['stage=smoke','num_envs='+('256' if args.benchmark else '211' if args.sweep else '3'),'rl_device='+('cuda:0' if args.benchmark else 'cpu'),'pipeline='+('gpu' if args.benchmark else 'cpu')])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='smoke';task['env']['asset']['handAsset']=f'assets/allegro/allegro_digit_{args.mode}.urdf'
env=AllegroDigitTwoAxis(task,'cuda:0',-1,True);start=time.perf_counter();report={}
try:
 
 if args.rollout:
  import yaml
  c=yaml.safe_load((ROOT/'configs/manip_official_sphere55.yaml').read_text())
  env.grasp_cache=torch.load(ROOT/c['task']['env']['grasp']['cache'],map_location='cpu')
 env.reset();names=env.gym.get_asset_rigid_body_names(env.hand_asset);dt=env.dt
 pose_names=list(POSTURES)
 if args.sweep:
  pose_records=json.loads((out/'sweep_compact_raw.json').read_text())['compact'][:211]
  pose_names=[str(i) for i in range(len(pose_records))]
 if args.benchmark:q=torch.tensor(POSTURES['opposition'],device=env.device)[None].repeat(256,1)
 elif args.sweep:q=torch.tensor([r['q'] for r in pose_records],device=env.device)
 else:q=torch.tensor([POSTURES[n] for n in pose_names],device=env.device)
 q=q[:,env.canonical_order];env.q.copy_(q);env.qdot.zero_();env.targets.copy_(q)
 env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_states))
 env.root_states[env.object_indices,0]=2;env.root_states[env.object_indices,2]=2;env.root_states[env.object_indices,7:]=0
 env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_states))
 if args.rollout:env.reset()  # Matched state fixtures, not a cross-asset grasp validation.
 initial_memory=None;contacts=[set() for _ in pose_names];max_force=[0.]*len(pose_names);gel_trigger=[0]*len(pose_names);frames=[];maxpenetration=[0.]*3
 rng=np.random.RandomState(14);timings=[]
 for step in range(args.steps):
  t=time.perf_counter()
  if args.benchmark and not args.rollout and step%60==0:
   target=env.lower+torch.tensor(rng.uniform(size=(256,16)),device=env.device,dtype=torch.float32)*(env.upper-env.lower)
   env.targets.copy_(target)
  
  if args.rollout:
   action=torch.ones((256,16),device=env.device)*(1 if (step//100)%2==0 else -1)
   env.step(action)
  else:
   env.update_low_level_control();env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True);env._refresh()
  if args.benchmark:torch.cuda.synchronize()
  if not torch.isfinite(env.q).all() or not torch.isfinite(env.contact_forces).all():raise RuntimeError('Nonfinite physics state')
  timings.append(time.perf_counter()-t)
  if not args.benchmark:
   for i,e in enumerate(env.envs):
    max_force[i]=max(max_force[i],float(env.contact_forces[i,:len(names)].norm(dim=-1).max()))
    gel_trigger[i]+=int((env.contact_forces[i,env.gel_ids].norm(dim=-1)>.1).any())
    for c in env.gym.get_env_rigid_contacts(e):
     a,b=int(c['body0']),int(c['body1'])
     if 0<=a<len(names) and 0<=b<len(names) and a!=b and float(c['lambda'])>0:contacts[i].add(tuple(sorted((names[a],names[b]))))
  if step%60==0:frames.append({'step':step,'torch_allocated_mib':torch.cuda.memory_allocated()/2**20,'torch_reserved_mib':torch.cuda.memory_reserved()/2**20,'process_mib':memory[-1]['mib'] if memory else None})
 report={'mode':args.mode,'asset_hash':env.asset_hash,'n_envs':env.num_envs,'steps':args.steps,'physical_seconds':args.steps*dt*(6 if args.rollout else 1),'elapsed_seconds':time.perf_counter()-start,'actual_rigid_shapes':env.gym.get_asset_rigid_shape_count(env.hand_asset),'torch_allocated_mib':torch.cuda.memory_allocated()/2**20,'torch_reserved_mib':torch.cuda.memory_reserved()/2**20,'samples':frames,'memory':memory,'peak_process_mib':max([m['mib'] for m in memory],default=None),'median_step_ms':float(np.median(timings[10:])*1000),'last60_mean_step_ms':float(np.mean(timings[-60:])*1000),'finite':True}
 if not args.benchmark:report['postures']={n:{'contact_pairs':sorted(contacts[i]),'max_force_N':max_force[i],'joint_error_rad':float((env.q[i]-q[i]).abs().max()),'gel_any_trigger_frame_fraction':gel_trigger[i]/args.steps} for i,n in enumerate(pose_names)}
 (out/(args.mode+('_sweep_physx' if args.sweep else '_rollout' if args.rollout else '_benchmark' if args.benchmark else '_physx')+'.json')).write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k not in ['memory','samples']},indent=2))
finally:stop.set();thread.join();env.close()
