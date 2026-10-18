"""Small integration check, not policy performance evaluation."""
import isaacgym
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.misc import set_seed

out = Path(__file__).resolve().parent
with initialize_config_dir(version_base='1.3', config_dir=str(ROOT/'configs')):
    cfg = compose(config_name='manip_tabletop_box45', overrides=[
        'num_envs=16', 'stage=teacher', 'pipeline=gpu', 'sim_device=cuda:0',
        'rl_device=cuda:0', 'graphics_device_id=-1', 'task.env.episodeLength=50'])
task = OmegaConf.to_container(cfg.task, resolve=True)
task['stage'] = 'teacher'
assert task['env']['manipulation']['pose_weight'] == .05
assert task['env']['manipulation']['contact_weight'] == .05
assert task['env']['manipulation']['fingertip_distance_weight'] == .1
assert task['env']['manipulation']['off_axis_weight'] == 0
set_seed(42)
env = AllegroDigitTabletop(task, 'cuda:0', -1, True)
checked = [0]
original = env.pre_physics_step

def check_pre(action):
    q = env.q.clone()
    previous_action = env.actions.clone()
    expected_action = .5*action.to(env.device) + .5*previous_action
    expected_target = (q + .026*expected_action).clamp(env.lower, env.upper)
    original(action)
    torch.testing.assert_close(env.targets, expected_target)
    expected_lost = (.026*expected_action - (expected_target-q)).abs().sum(-1)/(.026*expected_action).abs().sum(-1).clamp_min(1e-8)
    torch.testing.assert_close(env.target_increment_lost, expected_lost)
    checked[0] += 1

env.pre_physics_step = check_pre
try:
    obs = env.reset()
    reward_sum = 0.
    distance_sum = 0.
    completed = 0
    for step in range(100):
        action = .2*torch.sin(torch.arange(16, dtype=torch.float, device=env.device)[None,:] + step*.2).expand(16,-1)
        obs, reward, done, extra = env.step(action)
        assert all(torch.isfinite(v).all() for v in obs.values())
        assert torch.isfinite(reward).all()
        r = env.reward_terms['fingertip_distance']
        assert ((r >= 0) & (r <= .1)).all()
        contact = env.reward_terms['contact']
        assert ((contact == 0) | (contact == .05)).all()
        assert (env.reward_terms['pose_penalty'] <= 0).all()
        reward_sum += r.mean().item()
        distance_sum += extra['contact/gel_object_center_distance_m'].item()
        completed += done.sum().item()
    report = dict(passed=True,envs=16,steps=100,control_checks=checked[0],
                  action_reference=env.action_reference, resets=int(completed),
                  mean_distance_reward=reward_sum/100,mean_gel_object_distance_m=distance_sum/100,
                  obs_dim=env.num_obs, collision_sha256=env.collision_hash,
                  note='GPU PhysX integration; scripted actions, not learned rotation performance')
    (out/'simulation_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:
    env.close()
