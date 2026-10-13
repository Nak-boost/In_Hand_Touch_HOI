"""Hold diagnostic poses using the task PD controller, without policy or grasps."""
import argparse
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', choices=['all', 'open', 'half_closed', 'opposition'], default='opposition')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=int, default=0, help='Physics steps; 0 means until viewer exit')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 0 or (args.headless and not args.steps):
        parser.error('Headless inspection requires positive --steps')
    import isaacgym
    from isaacgym import gymtorch, gymutil, gymapi
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from train_manip import ROOT
    from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
    from hora.utils.mount_geometry import POSTURES
    names = list(POSTURES) if args.pose == 'all' else [args.pose]
    np.random.seed(0)
    torch.manual_seed(0)
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='manip', overrides=['stage=smoke', 'num_envs='+str(len(names)),
                                                     'rl_device=cpu', 'pipeline=cpu'])
    task = OmegaConf.to_container(cfg.task, resolve=True)
    task['stage'] = 'smoke'
    env = AllegroDigitTwoAxis(task, 'cuda:0', -1 if args.headless else 0, args.headless)
    contacts = [set() for _ in names]
    max_force = [0. for _ in names]
    steps = 0
    try:
        env.reset()
        q = torch.tensor([POSTURES[name] for name in names], dtype=torch.float)[:, env.canonical_order]
        env.q.copy_(q)
        env.qdot.zero_()
        env.targets.copy_(q)
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_states))
        env.root_states[env.object_indices, 0] = 2.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        bodies = env.gym.get_asset_rigid_body_names(env.hand_asset)
        axes = gymutil.AxesGeometry(.025)
        print('Diagnostic poses:', names, '| no policy/cache; object moved away | Esc exits', flush=True)
        for step in itertools.count():
            if args.steps and step >= args.steps:
                break
            env.update_low_level_control()
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env._refresh()
            if not torch.isfinite(env.q).all() or not torch.isfinite(env.contact_forces).all():
                raise FloatingPointError('Nonfinite diagnostic state')
            steps = step+1
            for i, e in enumerate(env.envs):
                max_force[i] = max(max_force[i], float(env.contact_forces[i, :len(bodies)].norm(dim=-1).max()))
                for c in env.gym.get_env_rigid_contacts(e):
                    a, b = int(c['body0']), int(c['body1'])
                    if 0 <= a < len(bodies) and 0 <= b < len(bodies) and a != b and float(c['lambda']) > 0:
                        contacts[i].add(tuple(sorted((bodies[a], bodies[b]))))
            if env.viewer:
                env.gym.clear_lines(env.viewer)
                for i, e in enumerate(env.envs):
                    for body in env.gel_ids:
                        state = env.bodies[i, body]
                        pose = gymapi.Transform()
                        pose.p = gymapi.Vec3(*state[:3].tolist())
                        pose.r = gymapi.Quat(*state[3:7].tolist())
                        gymutil.draw_lines(axes, env.gym, env.viewer, e, pose)
                env.render()
    finally:
        report = {'asset': task['env']['asset']['handAsset'], 'asset_sha256': env.asset_hash,
                  'physics_steps': steps, 'physics_dt': env.dt,
                  'method': 'Task limited-torque PD hold; object moved away; CPU tensor pipeline; GPU PhysX contacts',
                  'postures': {name: {'contact_pairs': sorted(contacts[i]), 'max_hand_force_N': max_force[i],
                                     'final_max_joint_error_rad': float((env.q[i]-env.targets[i]).abs().max())}
                               for i, name in enumerate(names)}}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
