"""Exercise actual PhysX hand contacts with and without self-collision masks.

Uses the CPU contact API for body-pair provenance, with no training or cache writes.
The object is moved away so any reported forces must arise from hand/hand contact.
"""
import isaacgym
from isaacgym import gymapi, gymtorch
import argparse
import faulthandler
import json
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from train_manip import ROOT  # registers the project's Hydra resolvers
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.utils.hand_collision import verify_actor_collisions


def run(disabled):
    print('Creating collision probe, disabled=', disabled, flush=True)
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='manip', overrides=['stage=smoke', 'num_envs=32',
                      'sim_device=cuda:0', 'rl_device=cpu', 'pipeline=cpu'])
    task = OmegaConf.to_container(cfg.task, resolve=True)
    task['stage'] = 'smoke'
    env = AllegroDigitTwoAxis(task, 'cuda:0', -1, True)
    print('Probe environment ready', flush=True)
    try:
        for e in env.envs:
            if disabled:
                shapes = env.gym.get_actor_rigid_shape_properties(e, 0)
                for shape in shapes:
                    shape.filter = 1
                env.gym.set_actor_rigid_shape_properties(e, 0, shapes)
            else:
                verify_actor_collisions(env.gym, e, 0, env.collision_spec)
        names = env.gym.get_asset_rigid_body_names(env.hand_asset)
        excluded = set(map(tuple, env.collision_spec['excluded_pairs']))
        pairs, forbidden = set(), set()
        max_force = 0.
        # Several deterministic starting postures, including closure/crossing.
        for trial in range(4):
            print('Collision probe trial', trial, flush=True)
            torch.manual_seed(100+trial)
            np.random.seed(100+trial)
            env.reset()
            env.root_states[env.object_indices, 2] = 2.0
            env.root_states[env.object_indices, 7:] = 0
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
            target = env.lower + torch.rand_like(env.q)*(env.upper-env.lower)
            env.targets.copy_(target)
            for step in range(35):
                env.update_low_level_control()
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
                env._refresh()
                assert torch.isfinite(env.q).all() and torch.isfinite(env.contact_forces).all()
                max_force = max(max_force, env.contact_forces[:, :len(names)].norm(dim=-1).max().item())
                for e in env.envs:
                    for c in env.gym.get_env_rigid_contacts(e):
                        a, b = int(c['body0']), int(c['body1'])
                        if 0 <= a < len(names) and 0 <= b < len(names) and a != b:
                            pair = tuple(sorted((names[a], names[b])))
                            pairs.add(pair)
                            if pair in excluded:
                                forbidden.add(pair)
        return dict(disabled=disabled, hand_contact_pairs=sorted(pairs),
                    forbidden_pairs=sorted(forbidden), max_hand_contact_force=max_force)
    finally:
        env.close()


if __name__ == '__main__':
    faulthandler.dump_traceback_later(30, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    enabled = run(False)
    disabled = run(True)
    assert enabled['hand_contact_pairs'], 'Enabled hand produced no self contacts'
    assert not enabled['forbidden_pairs'], 'Excluded assembly contacts were reported'
    assert not disabled['hand_contact_pairs'], 'Disabled comparison unexpectedly collided'
    result = dict(enabled=enabled, disabled=disabled)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
