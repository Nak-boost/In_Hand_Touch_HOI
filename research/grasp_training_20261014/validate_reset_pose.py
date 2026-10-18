import isaacgym
import json
from pathlib import Path
import torch
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
 cfg=compose(config_name='manip_official_box45',overrides=['stage=teacher','sim_device=cpu','rl_device=cpu','pipeline=cpu','num_envs=4','graphics_device_id=-1','task.env.grasp.cache=cache/digit_box45_official_candidates_20261014.pt'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher';env=AllegroDigitTwoAxis(task,'cpu',-1,True)
try:
 env.reset();env.stage='grasps';torch.testing.assert_close(env.initial_q,env.q)
 bank=env.grasp_cache['states']['box45'];targets=env.grasp_cache['targets']['box45']
 for i in range(4):
  index=(bank[:,:16]-env.q[i]).abs().sum(-1).argmin();torch.testing.assert_close(env.targets[i],targets[index])
 before=env.initial_q.clone()
 for _ in range(8):env.step(torch.zeros_like(env.actions))
 torch.testing.assert_close(env.initial_q,before)
 expected=-.3*(env.q-env.initial_q).square().sum(-1).mean()
 torch.testing.assert_close(env.extras['reward/pose_penalty'],expected)
 env.reset_idx(torch.tensor([1,3]));torch.testing.assert_close(env.initial_q[[0,2]],before[[0,2]])
 torch.testing.assert_close(env.initial_q[[1,3]],env.q[[1,3]])
 del env.grasp_cache['targets'];env.reset();torch.testing.assert_close(env.targets,env.q)
 torch.testing.assert_close(env.initial_q,env.q)
 report=dict(paired_targets_restored=True,reference_stable_during_steps=True,partial_reset_isolated=True,legacy_cache_fallback=True,pose_reward_matches_formula=True,physics='CPU PhysX; behavior checks only')
 Path(__file__).with_name('reset_pose_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
finally:env.close()
