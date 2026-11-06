"""Load newly selected banks, instantiate restored network, check table reward gate."""
import isaacgym
import argparse,json,pathlib
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.algo.models.models import ActorCritic
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.manip_control import tracking_action
p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();out=pathlib.Path(__file__).parent
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):cfg=compose(config_name=a.config)
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher';task['env']['numEnvs']=max(6,len(task['env']['objects']));task['rl_device']='cpu'
torch.set_num_threads(1);torch.manual_seed(771);np.random.seed(771)
cls=AllegroDigitTabletop if task['name']=='AllegroDigitTabletop' else AllegroDigitTwoAxis
env=cls(task,'cuda:0',-1,True)
try:
 assert list(cfg.train.network.mlp.units)==[256,256,128]
 assert list(cfg.train.network.priv_mlp.units)==[128,64,8]
 model=ActorCritic(network_spec(cfg,env.num_obs)).eval()
 obs=env.reset();target=env.targets.clone()
 for step in range(10):
  with torch.no_grad():pred=model.act_inference(obs)
  assert pred.shape==(env.num_envs,16) and torch.isfinite(pred).all()
  obs,reward,done,_=env.step(tracking_action(env.q,env.targets,target,env.actions,env.cfg))
  assert torch.isfinite(reward).all()
  raw=env.reward_diagnostics['rotation_unconditioned']
  count=env.task_gel_contacts.long().sum(-1).clamp_max(2)
  multiplier=torch.tensor([.1,.5,1.])[count]
  expected=raw.clamp_min(0)*multiplier+raw.clamp_max(0)
  assert torch.allclose(env.reward_terms['rotation'],expected)
 result=dict(config=a.config,cache=str(env.cache_path),asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,control_dt=env.control_dt,network=list(cfg.train.network.mlp.units),priv_network=list(cfg.train.network.priv_mlp.units),output_shape=list(pred.shape),contact_gate_verified=True,steps=10)
 (out/(a.config+'_loadcheck.json')).write_text(json.dumps(result,indent=2));print('LOAD_AND_GATE_PASS',a.config,flush=True)
finally:env.close();print('CLOSED',flush=True)
