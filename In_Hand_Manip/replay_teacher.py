"""Replay saved privileged teachers without training or changing task physics."""
import argparse
import itertools
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=['sphere55', 'box50', 'capsule70'], default='sphere55')
    parser.add_argument('--run-dir', type=Path, help='Override the default saved experiment directory')
    parser.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=0, help='0: keep displaying until Esc or Ctrl-C')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    parser.add_argument('--sampled', action='store_true', help='Sample Gaussian actions instead of using the mean')
    parser.add_argument('--headless', action='store_true', help='Check inference without a window; requires --steps')
    parser.add_argument('--no-axes', action='store_true', help='Hide the visual object-orientation axes')
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0 or (args.headless and args.steps == 0):
        parser.error('Require positive --num-envs, nonnegative --steps, and finite steps for headless replay')

    # Required import order for Isaac Gym Preview 4.
    import isaacgym
    from isaacgym import gymapi, gymutil
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hora.algo.models.models import ActorCritic
    from hora.algo.models.running_mean_std import RunningMeanStd
    from hora.algo.padapt.tactile_distill import network_spec
    from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
    from hora.utils.hand_collision import require_collision_signature

    root = Path(__file__).resolve().parent
    run = args.run_dir.expanduser().resolve() if args.run_dir else (
        root.parent / 'runs/manip' / ('teacher_' + args.object + '_z_self_collision_v1'))
    config_path = run / 'config.yaml'
    checkpoint_path = run / 'stage1_nn' / (args.checkpoint + '.pth')
    for path in [config_path, checkpoint_path]:
        if not path.is_file():
            raise FileNotFoundError('Copy the teacher replay bundle first; missing: ' + str(path))
    cfg = OmegaConf.load(config_path)
    # Saved configurations are fully resolved. Replay uses their controller,
    # contact, randomization and reward settings, including automatic resets.
    task = OmegaConf.to_container(cfg.task, resolve=True)
    task['stage'] = 'teacher'
    task['env']['numEnvs'] = args.num_envs
    task['rl_device'] = args.device
    cache = root / task['env']['grasp']['cache']
    if not cache.is_file():
        raise FileNotFoundError('Copy the matching saved grasp cache: ' + str(cache))
    for relative in ['hora/tasks/allegro_digit_two_axis.py', 'hora/tasks/manipulation_math.py',
                     'hora/algo/models/models.py', 'hora/algo/models/running_mean_std.py']:
        snapshot = run / 'source' / relative
        if snapshot.is_file() and snapshot.read_bytes() != (root / relative).read_bytes():
            raise ValueError('Inference source differs from training snapshot: ' + relative)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('kind') != 'privileged_teacher':
        raise ValueError('This viewer requires a privileged teacher checkpoint')
    env = AllegroDigitTwoAxis(task, args.device, args.graphics_device_id, args.headless)
    try:
        if checkpoint.get('asset_sha256') != env.asset_hash:
            raise ValueError('Teacher hand asset mismatch')
        require_collision_signature(checkpoint, env.collision_hash, 'Teacher')
        model = ActorCritic(network_spec(cfg, env.num_obs)).to(env.device).eval()
        model.load_state_dict(checkpoint['model'])
        normalizer = RunningMeanStd((env.num_obs,)).to(env.device).eval()
        normalizer.load_state_dict(checkpoint['running_mean_std'])
        axes = gymutil.AxesGeometry(0.045)
        obs = env.reset()
        print('PRIVILEGED TEACHER REPLAY:', checkpoint_path, flush=True)
        print('Objects:', [obj['name'] for obj in env.catalog],
              '| actions:', 'sampled' if args.sampled else 'mean',
              '| Esc / Ctrl-C: exit | RGB axes: object local X/Y/Z', flush=True)
        completed = failed = 0
        with torch.no_grad():
            for step in itertools.count():
                if args.steps and step >= args.steps:
                    break
                if env.viewer and not args.no_axes:
                    env.gym.clear_lines(env.viewer)
                    for index, state in enumerate(env.object_state[:, :7].cpu().tolist()):
                        pose = gymapi.Transform()
                        pose.p = gymapi.Vec3(*state[:3])
                        pose.r = gymapi.Quat(*state[3:7])
                        gymutil.draw_lines(axes, env.gym, env.viewer, env.envs[index], pose)
                inputs = {'obs': normalizer(obs['obs']) if cfg.train.ppo.normalize_input else obs['obs'],
                          'priv_info': obs['priv_info']}
                mu, logstd, _, _, _ = model._actor_critic(inputs)
                action = mu + logstd.exp() * torch.randn_like(mu) if args.sampled else mu
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Teacher produced nonfinite actions')
                obs, reward, _, _ = env.step(action.clamp(-1, 1))
                if not torch.isfinite(reward).all():
                    raise FloatingPointError('Simulation produced nonfinite rewards')
                for index in env.terminal['done'].nonzero(as_tuple=False).flatten().tolist():
                    completed += 1
                    failed += int(env.terminal['failed'][index])
                    turns = (env.terminal['angles'][index] / (2 * math.pi)).tolist()
                    print('Episode env=%d seconds=%.2f net_turns=%s failed=%s' % (
                        index, float(env.terminal['length'][index]) * env.control_dt,
                        [round(x, 3) for x in turns], bool(env.terminal['failed'][index])), flush=True)
                if (step + 1) % 100 == 0:
                    print('Simulated %.1fs | env0 current net turns: %s' % (
                        (step + 1) * env.control_dt,
                        [round(x, 3) for x in (env.total_angle[0] / (2 * math.pi)).tolist()]), flush=True)
        print('Replay complete: episodes=%d failed=%d' % (completed, failed), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
