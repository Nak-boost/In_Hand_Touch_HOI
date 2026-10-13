import isaacgym
import torch,json
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
 cfg=compose(config_name='manip',overrides=['num_envs=64'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
env=AllegroDigitTwoAxis(task,'cuda:0',-1,True)
try:
 torch.manual_seed(1)
 env.reset()
 env.stage='grasps'
 alive=torch.ones(64,dtype=torch.bool,device=env.device)
 stats=[]
 for step in range(100):
  _,_,_,_=env.step(torch.zeros_like(env.actions))
  alive &= ~env.terminal['failed']
  if step in [0,19,39,79,99]:
   stats.append(dict(seconds=(step+1)*env.control_dt,retained=int(alive.sum()),mean_gel_contacts=env.raw_contacts.sum(-1).mean().item()))
 print(json.dumps(stats))
 open('/tmp/collision_grasp_replay.json','w').write(json.dumps(stats,indent=2))
finally:
 env.close()
