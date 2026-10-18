"""Replay saved teacher or student policies without training or changing task physics."""
import argparse
import itertools
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=['sphere55', 'box50', 'capsule70'], default='sphere55')
    parser.add_argument('--run-dir', type=Path, help='Saved experiment: absolute path or relative to the project root')
    parser.add_argument('--checkpoint', default='last', help='best, last, or a checkpoint filename/path relative to the run')
    parser.add_argument('--num-envs', type=int, help='Default: one environment per object in the saved catalog')
    parser.add_argument('--steps', type=int, default=0, help='0: keep displaying until Esc or Ctrl-C')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    parser.add_argument('--sampled', action='store_true', help='Sample Gaussian actions instead of using the mean')
    parser.add_argument('--headless', action='store_true', help='Check inference without a window; requires --steps')
    parser.add_argument('--no-axes', action='store_true', help='Hide the visual object-orientation axes')
    parser.add_argument('--grasp-cache', type=Path, help='Override relocated cache path; signatures still checked')
    parser.add_argument('--allow-source-mismatch', action='store_true',
                        help='Explicitly allow replay after inference source edits; physics/model signatures still checked')
    args = parser.parse_args()
    if (args.num_envs is not None and args.num_envs < 1) or args.steps < 0 or (args.headless and args.steps == 0):
        parser.error('Require positive --num-envs, nonnegative --steps, and finite steps for headless replay')

    # Required import order for Isaac Gym Preview 4.
    import isaacgym
    from isaacgym import gymapi, gymutil
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hora.algo.models.models import ActorCritic
    from hora.algo.models.tactile_student import TactileStudent
    from hora.utils.replay_files import resolve_checkpoint
    from hora.algo.models.running_mean_std import RunningMeanStd
    from hora.algo.padapt.tactile_distill import network_spec, normalize_inputs
    from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
    from hora.utils.hand_collision import require_collision_signature

    root = Path(__file__).resolve().parent
    run = (root.parent / args.run_dir.expanduser()).resolve() if args.run_dir else (
        root.parent / 'runs/manip' / ('teacher_' + args.object + '_z_self_collision_v1'))
    config_path = run / 'config.yaml'
    checkpoint_path = resolve_checkpoint(run, args.checkpoint)
    for path in [config_path, checkpoint_path]:
        if not path.is_file():
            raise FileNotFoundError('Sync the saved experiment first; missing: ' + str(path))
    cfg = OmegaConf.load(config_path)
    # Saved configurations are fully resolved. Replay uses their controller,
    # contact, randomization and reward settings, including automatic resets.
    task = OmegaConf.to_container(cfg.task, resolve=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    kind = checkpoint.get('kind')
    if kind not in ('privileged_teacher', 'tactile_student'):
        raise ValueError('Unsupported checkpoint kind: ' + str(kind))
    student = kind == 'tactile_student'
    task['stage'] = 'student' if student else 'teacher'
    task['env']['numEnvs'] = args.num_envs or len(task['env']['objects'])
    if task['env']['numEnvs'] < len(task['env']['objects']):
        parser.error('--num-envs must cover every object in the saved catalog')
    if args.grasp_cache:
        task['env']['grasp']['cache'] = str(args.grasp_cache.expanduser().resolve())
    task['rl_device'] = args.device
    cache = root / task['env']['grasp']['cache']
    if not cache.is_file():
        raise FileNotFoundError('Copy the matching saved grasp cache: ' + str(cache))
    source_files = ['hora/tasks/allegro_digit_two_axis.py', 'hora/tasks/manipulation_math.py',
                    'hora/algo/models/models.py', 'hora/algo/models/running_mean_std.py',
                    'hora/tasks/base/vec_task.py', 'hora/utils/hand_collision.py',
                    'hora/algo/padapt/tactile_distill.py']
    if student:
        source_files.append('hora/algo/models/tactile_student.py')
    mismatches, missing = [], []
    for relative in source_files:
        snapshot = run / 'source' / relative
        if snapshot.is_file() and snapshot.read_bytes() != (root / relative).read_bytes():
            mismatches.append(relative)
        if not snapshot.is_file():
            missing.append(relative)
    if mismatches and not args.allow_source_mismatch:
        raise ValueError('Inference source differs from training snapshot: ' + ', '.join(mismatches)
                         + '. Use the matching code revision or explicitly pass --allow-source-mismatch.')
    if mismatches:
        print('SOURCE MISMATCH OVERRIDE:', mismatches, flush=True)
    if missing:
        print('Source snapshots unavailable for:', missing, flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = AllegroDigitTwoAxis(task, args.device, args.graphics_device_id, args.headless)
    try:
        if checkpoint.get('asset_sha256') != env.asset_hash:
            raise ValueError('Policy hand asset mismatch')
        require_collision_signature(checkpoint, env.collision_hash, 'Policy')
        if student:
            if checkpoint['history_length'] != env.prop_hist_len or checkpoint['history_dim'] != env.history_dim:
                raise ValueError('Student history configuration mismatch')
            if checkpoint['joint_names'] != env.joint_names:
                raise ValueError('Student joint order mismatch')
            for key in ['action_scale', 'action_smoothing']:
                if checkpoint[key] != env.cfg[key]:
                    raise ValueError('Student controller mismatch: ' + key)
            if not torch.allclose(checkpoint['lower'].to(env.device), env.lower) or not torch.allclose(checkpoint['upper'].to(env.device), env.upper):
                raise ValueError('Student joint limits mismatch')
            model = TactileStudent(checkpoint['network'], checkpoint['history_dim'])
        else:
            model = ActorCritic(network_spec(cfg, env.num_obs))
        model = model.to(env.device).eval()
        model.load_state_dict(checkpoint['model'])
        normalizer = RunningMeanStd((env.num_obs,)).to(env.device).eval()
        normalizer.load_state_dict(checkpoint['running_mean_std'])
        axes = gymutil.AxesGeometry(0.045)
        obs = env.reset()
        print('TACTILE STUDENT REPLAY:' if student else 'PRIVILEGED TEACHER REPLAY:', checkpoint_path, flush=True)
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
                if student:
                    # Never pass object truth to the student's policy interface.
                    inputs = normalize_inputs(normalizer, obs['obs'], obs['proprio_hist'])
                else:
                    inputs = {'obs': normalizer(obs['obs']) if cfg.train.ppo.normalize_input else obs['obs'],
                              'priv_info': obs['priv_info']}
                mu, logstd, _, _, _ = model._actor_critic(inputs)
                action = mu + logstd.exp() * torch.randn_like(mu) if args.sampled else mu
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Policy produced nonfinite actions')
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
