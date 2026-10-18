"""Project entry point. Run with scripts/run_manip.sh for the local Isaac Gym environment."""
import isaacgym  # Isaac Gym must be imported before torch
import datetime
import faulthandler
import json
import os
import signal
import subprocess
import traceback
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from hora.algo.ppo.ppo import PPO
from hora.algo.padapt.tactile_distill import TactileDistill, normalize_inputs
from hora.algo.models.tactile_student import TactileStudent
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import action_reference, require_action_reference
from hora.utils.misc import set_seed
from hora.utils.training_watchdog import TrainingJournal, atomic_torch_save

JOURNAL = TrainingJournal(os.environ.get('MANIP_HEARTBEAT'))

ROOT = Path(__file__).resolve().parent
for name, resolver in {
    'eq': lambda x, y: str(x).lower() == str(y).lower(),
    'contains': lambda x, y: x.lower() in y.lower(),
    'if': lambda pred, a, b: a if pred else b,
    'resolve_default': lambda default, arg: default if arg == '' else arg,
}.items():
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)


class ManipPPO(PPO):
    def play_steps(self):
        JOURNAL.write('collect', agent_steps=self.agent_steps, epoch_num=self.epoch_num)
        result = super().play_steps()
        JOURNAL.write('update', agent_steps=self.agent_steps, epoch_num=self.epoch_num)
        return result

    def train_epoch(self):
        result = super().train_epoch()
        JOURNAL.write('rollout_complete', agent_steps=self.agent_steps, epoch_num=self.epoch_num)
        return result

    def save(self, name):
        JOURNAL.write('checkpoint_save', agent_steps=self.agent_steps, epoch_num=self.epoch_num)
        ckpt = dict(model=self.model.state_dict(),
                    running_mean_std=self.running_mean_std.state_dict(),
                    value_mean_std=self.value_mean_std.state_dict())
        ckpt.update(asset_sha256=self.env.asset_hash, kind='privileged_teacher',
                    action_reference=action_reference(self.env.cfg),
                    collision_sha256=self.env.collision_hash,
                    optimizer=self.optimizer.state_dict(), agent_steps=self.agent_steps,
                    epoch_num=self.epoch_num, best_rewards=self.best_rewards, last_lr=self.last_lr)
        atomic_torch_save(ckpt, str(name)+'.pth')
        JOURNAL.write('checkpoint_saved', checkpoint=str(name)+'.pth',
                      agent_steps=self.agent_steps, epoch_num=self.epoch_num)

    def restore_train(self, fn):
        if not fn:
            return
        ckpt = torch.load(fn, map_location=self.device)
        require_action_reference(ckpt, self.env.cfg, 'Teacher checkpoint')
        require_collision_signature(ckpt, self.env.collision_hash, 'Teacher checkpoint')
        if ckpt.get('asset_sha256') != self.env.asset_hash:
            raise ValueError('Checkpoint is not from this DIGIT asset')
        super().restore_train(fn)
        self.value_mean_std.load_state_dict(ckpt['value_mean_std'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.agent_steps = ckpt['agent_steps']
            self.epoch_num = ckpt['epoch_num']
            self.last_lr = ckpt.get('last_lr', self.optimizer.param_groups[0]['lr'])
            self.best_rewards = ckpt.get('best_rewards', -10000)


@torch.no_grad()
def evaluate(env, cfg, run_dir):
    if not cfg.student_checkpoint:
        raise ValueError('Evaluation requires student_checkpoint; privileged teacher is not evaluated as deployable')
    ckpt = torch.load(cfg.student_checkpoint, map_location=cfg.rl_device)
    require_action_reference(ckpt, env.cfg, 'Student checkpoint')
    require_collision_signature(ckpt, env.collision_hash, 'Student checkpoint')
    if ckpt.get('kind') != 'tactile_student' or ckpt['asset_sha256'] != env.asset_hash:
        raise ValueError('Expected compatible tactile student checkpoint')
    if ckpt['history_length'] != env.prop_hist_len:
        raise ValueError('History length mismatch')
    for key in ['action_scale', 'action_smoothing']:
        if ckpt[key] != env.cfg[key]:
            raise ValueError('Deployment controller mismatch: '+key)
    model = TactileStudent(ckpt['network'], ckpt['history_dim']).to(cfg.rl_device).eval()
    model.load_state_dict(ckpt['model'])
    rms = RunningMeanStd((env.num_obs,)).to(cfg.rl_device).eval()
    rms.load_state_dict(ckpt['running_mean_std'])
    obs = env.reset()
    records = []
    for _ in range(cfg.evaluation_steps):
        # No privileged key is passed to the deployable actor.
        action = model.act_inference(normalize_inputs(rms, obs['obs'], obs['proprio_hist'])).clamp(-1, 1)
        obs, _, _, _ = env.step(action)
        for index in env.terminal['done'].nonzero(as_tuple=False).flatten().tolist():
            records.append(dict(object=env.catalog[env.object_ids[index].item()]['name'],
                                failed=env.terminal['failed'][index].item(),
                                duration=env.terminal['length'][index].item()*env.control_dt,
                                signed_radians_per_axis=env.terminal['angles'][index].tolist()))
    successful = [not r['failed'] and all(a >= 2*3.141592653589793 for a in r['signed_radians_per_axis']) for r in records]
    report = dict(completed_episodes=len(records), episodes=records,
                  commanded_axes=env.axis_options.cpu().tolist(),
                  retention_rate=sum(not r['failed'] for r in records)/len(records) if records else None,
                  all_axes_one_turn_rate=sum(successful)/len(records) if records else None,
                  partial_signed_radians_per_axis=env.total_angle.cpu().tolist(),
                  note='Angles use simulation truth for evaluation only; no precise-turn stopping is implemented.')
    if len(env.axis_options) == 2:
        report['both_axes_one_turn_rate'] = report['all_axes_one_turn_rate']
    (run_dir/'evaluation.json').write_text(json.dumps(report, indent=2))
    print('Evaluation completed episodes:', len(records))


@hydra.main(version_base='1.3', config_path='configs', config_name='manip')
def main(cfg):
    if JOURNAL.path:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    JOURNAL.write('initializing')
    set_seed(cfg.seed)
    if cfg.stage not in ('teacher', 'student', 'evaluate', 'grasps', 'smoke'):
        raise ValueError('Unknown stage: '+cfg.stage)
    if cfg.stage == 'student' and not cfg.teacher_checkpoint:
        raise ValueError('student stage requires teacher_checkpoint')
    if cfg.stage == 'teacher' and cfg.train.ppo.minibatch_size > int(cfg.num_envs)*cfg.train.ppo.horizon_length:
        raise ValueError('Reduce train.ppo.minibatch_size for small num_envs')
    run_name = cfg.run_name or (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+cfg.stage)
    run_dir = (ROOT/cfg.run_root/run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (run_dir/'git_revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode())
    (run_dir/'gitdiff.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    # Include new, untracked implementation files too; git diff alone omits them.
    import shutil
    for directory in ['hora', 'configs', 'scripts']:
        shutil.copytree(ROOT/directory, run_dir/'source'/directory,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, run_dir/'source'/'train_manip.py')
    task = OmegaConf.to_container(cfg.task, resolve=True)
    task['stage'] = cfg.stage
    task_classes = {'AllegroDigitTwoAxis': AllegroDigitTwoAxis, 'AllegroDigitTabletop': AllegroDigitTabletop}
    env = task_classes[task['name']](task, cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    JOURNAL.write('environment_ready')
    try:
        (run_dir/'asset_metadata.json').write_text(json.dumps(dict(
            asset_sha256=env.asset_hash, collision_sha256=env.collision_hash,
            collision_spec=env.collision_spec, joint_names=env.joint_names, fingers=['index', 'middle', 'ring', 'thumb'],
            control_dt=env.control_dt, action_reference=action_reference(env.cfg), objects=env.catalog), indent=2))
        shutil.copy2(ROOT/task['env']['asset']['handAsset'], run_dir/'allegro_digit.urdf')
        if cfg.stage == 'grasps':
            env.generate_grasps()
        elif cfg.stage == 'teacher':
            agent = ManipPPO(env, str(run_dir), cfg)
            agent.restore_train(cfg.checkpoint)
            try:
                agent.train()
                agent.save(str(run_dir/'stage1_nn'/'last'))
                JOURNAL.write('training_complete', training_complete=True,
                              agent_steps=agent.agent_steps, epoch_num=agent.epoch_num)
            finally:
                agent.writer.close()
        elif cfg.stage == 'student':
            TactileDistill(env, run_dir, cfg).train()
        elif cfg.stage == 'evaluate':
            evaluate(env, cfg, run_dir)
        else:
            obs = env.reset()
            for _ in range(cfg.evaluation_steps):
                obs, reward, done, _ = env.step(torch.zeros(env.num_envs, 16, device=cfg.rl_device))
                if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(reward).all():
                    raise FloatingPointError('Smoke test produced nonfinite observations/rewards')
            print('SMOKE PASS', {k: list(v.shape) for k, v in obs.items()})
    except BaseException:
        JOURNAL.write('exception', exception=traceback.format_exc())
        raise
    finally:
        JOURNAL.write('destroy_sim')
        env.close()
    JOURNAL.write('closed')
    print('Run directory:', run_dir)


if __name__ == '__main__':
    main()
