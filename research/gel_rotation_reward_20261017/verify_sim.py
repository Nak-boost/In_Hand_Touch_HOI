"""Verify experiment B on real contact trajectories; no policy optimization."""
import isaacgym
import json
from pathlib import Path
import torch
import numpy as np
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import require_action_reference
out=Path(__file__).resolve().parent
np.random.seed(215);torch.manual_seed(215);torch.set_num_threads(1)
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
 cfg=compose(config_name='manip_hora16384_sphere55_gel_rotation',overrides=['num_envs=32','rl_device=cpu'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
(out/'config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
env=AllegroDigitTwoAxis(task,'cuda:0',-1,True)
try:
 path=ROOT.parent/'runs/manip/teacher_sphere55_z_hora16384_pairfilter_20261016/stage1_nn/best.pth'
 ck=torch.load(path,map_location='cpu')
 require_collision_signature(ck,env.collision_hash,'Verification policy')
 require_action_reference(ck,env.cfg,'Verification policy')
 assert ck['asset_sha256']==env.asset_hash
 model=ActorCritic(network_spec(cfg,env.num_obs)).eval();model.load_state_dict(ck['model'])
 rms=RunningMeanStd((env.num_obs,)).eval();rms.load_state_dict(ck['running_mean_std'])
 obs=env.reset();counts=torch.zeros(3,dtype=torch.long);signs=torch.zeros(3,dtype=torch.long)
 means={'old_rotation':0.,'new_rotation':0.,'old_total_counterfactual':0.,'new_total':0.};maxerror=0.
 for step in range(240):
  with torch.no_grad():
   mu,*_=model._actor_critic({'obs':rms(obs['obs']),'priv_info':obs['priv_info']})
   obs,rew,_,_=env.step(mu.clamp(-1,1))
  assert torch.isfinite(rew).all()
  n=env.task_gel_contacts.sum(-1);group=n.clamp_max(2)
  scale=torch.where(n==0,.1,torch.where(n==1,.5,1.))
  raw=env.reward_diagnostics['rotation_unconditioned']
  expected=torch.where(raw>0,scale*raw,raw)
  actual=env.reward_terms['rotation']
  torch.testing.assert_close(actual,expected)
  torch.testing.assert_close(env.reward_diagnostics['rotation_contact_scale'],scale)
  torch.testing.assert_close(rew,sum(env.reward_terms.values()))
  torch.testing.assert_close(env.reward_terms['contact'],.05*(n>=2))
  maxerror=max(maxerror,float((actual-expected).abs().max()))
  counts+=torch.bincount(group,minlength=3)
  signs+=torch.tensor([int((raw<0).sum()),int((raw==0).sum()),int((raw>0).sum())])
  means['old_rotation']+=float(raw.mean());means['new_rotation']+=float(actual.mean())
  means['old_total_counterfactual']+=float((rew+raw-actual).mean());means['new_total']+=float(rew.mean())
 assert (counts>0).all() and signs[0]>0 and signs[2]>0
 assert env.cfg['position_weight']==2 and env.cfg['max_displacement']==.13
 report=dict(steps=240,num_envs=32,environment_steps=7680,verified=True,max_rotation_error=maxerror,
  contact_count_frames_0_1_2plus=counts.tolist(),negative_zero_positive_frames=signs.tolist(),
  means={k:v/240 for k,v in means.items()},policy_checkpoint=str(path),
  note='Frozen old policy, new reward evaluated on actual filtered contacts; counterfactual old reward on identical frames. No optimizer updates; no evidence yet of better learned behavior.')
 (out/'simulation_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
finally:
 print('Entering simulation cleanup',flush=True)
 env.close()
 print('Simulation closed normally',flush=True)
