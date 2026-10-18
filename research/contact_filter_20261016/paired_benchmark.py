import isaacgym
import json,os,time
from pathlib import Path
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.object_contacts import GelObjectContactReader
from hora.utils.manip_control import tracking_action
out=Path(__file__).resolve().parent
np.random.seed(215);torch.manual_seed(215);torch.set_num_threads(1)
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
 cfg=compose(config_name='manip_tabletop_box45',overrides=['num_envs=256','pipeline=cpu','rl_device=cuda:0','graphics_device_id=-1'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
env=AllegroDigitTabletop(task,'cuda:0',-1,True);env.reset();env.stage='grasps'
reader=GelObjectContactReader(env);targets=env.targets.clone();durations={False:[],True:[]};querytimes=[]
refresh=reader.refresh
qtime=[0.]
def timed(e):
 start=time.perf_counter();r=refresh(e);qtime[0]=time.perf_counter()-start;return r
reader.refresh=timed
for step in range(500):
 enabled=step>=100 and ((step-100)%4 in (1,2))
 env.object_contact_reader=reader if enabled else None
 action=tracking_action(env.q,env.targets,targets,env.actions,env.cfg).cuda()
 torch.cuda.synchronize();start=time.perf_counter();env.step(action);torch.cuda.synchronize()
 elapsed=time.perf_counter()-start
 if step>=100:
  durations[enabled].append(elapsed)
  if enabled:querytimes.append(qtime[0])
 if (step+1)%100==0:print('step',step+1,flush=True)
# Exact observation invariance at one identical state, including noise and queue contents.
saved={k:getattr(env,k).clone() for k in ['contact_queue','proprio_hist_buf','at_reset_buf']}
rng=torch.get_rng_state()
env.object_contact_filter=False;env.object_contact_reader=None;env._observe()
off={k:v.clone() for k,v in env._observations().items()}
for k,v in saved.items():getattr(env,k).copy_(v)
torch.set_rng_state(rng)
env.object_contact_filter=True;env.object_contact_reader=reader;env._observe()
assert all(torch.equal(v,env._observations()[k]) for k,v in off.items())
r={'num_envs':256,'warmup_steps':100,'sequence':'ABBA, repeated 100 times, same physics trajectory and target-tracking actions','observation_invariance_at_fixed_state':True,'modes':{}}
for mode,values in durations.items():
 r['modes']['on' if mode else 'off']={'steps':len(values),'mean_ms':float(np.mean(values)*1000),'median_ms':float(np.median(values)*1000),'p95_ms':float(np.percentile(values,95)*1000),'env_steps_per_second':256/float(np.mean(values))}
r['query_mean_ms']=float(np.mean(querytimes)*1000);r['extra_step_time_fraction']=r['modes']['on']['mean_ms']/r['modes']['off']['mean_ms']-1
(out/'paired_benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);os._exit(0)
