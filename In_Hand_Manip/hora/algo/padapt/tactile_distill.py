"""Student-state rollouts, frozen privileged teacher, joint actor/TCN distillation."""
from pathlib import Path
import torch
from tensorboardX import SummaryWriter

from hora.algo.models.models import ActorCritic
from hora.algo.models.tactile_student import TactileStudent, distillation_loss
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import action_reference, require_action_reference


def network_spec(config, observation_dim, actions=16):
    return dict(actor_units=list(config.train.network.mlp.units),
                priv_mlp_units=list(config.train.network.priv_mlp.units), actions_num=actions,
                input_shape=(observation_dim,), priv_info=True, proprio_adapt=False,
                priv_info_dim=config.train.ppo.priv_info_dim)


def normalize_inputs(normalizer, obs, history):
    shape = history.shape
    return {'obs': normalizer(obs),
            'proprio_hist': normalizer(history.reshape(-1, shape[-1])).reshape(shape)}


class TactileDistill:
    def __init__(self, env, run_dir, config):
        self.env, self.cfg = env, config
        self.device = config.rl_device
        self.run_dir = Path(run_dir)
        self.spec = network_spec(config, env.num_obs)
        self.teacher = ActorCritic(dict(self.spec)).to(self.device).eval()
        checkpoint = torch.load(config.teacher_checkpoint, map_location=self.device)
        require_action_reference(checkpoint, env.cfg, 'Teacher checkpoint')
        require_collision_signature(checkpoint, env.collision_hash, 'Teacher checkpoint')
        if checkpoint.get('asset_sha256') != env.asset_hash:
            raise ValueError('Teacher checkpoint must be from this DIGIT asset, not upstream HORA')
        self.teacher.load_state_dict(checkpoint['model'])
        self.teacher.requires_grad_(False)
        self.normalizer = RunningMeanStd((env.num_obs,)).to(self.device).eval()
        self.normalizer.load_state_dict(checkpoint['running_mean_std'])
        self.student = TactileStudent(self.spec, env.history_dim).to(self.device)
        missing, unexpected = self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        if unexpected or any(not name.startswith('adapt_tconv.') for name in missing):
            raise ValueError('Teacher/student architecture mismatch')
        self.optimizer = torch.optim.Adam((p for p in self.student.parameters() if p.requires_grad), lr=config.distill.learning_rate)
        self.start_step = 0
        if config.student_checkpoint:
            ckpt = torch.load(config.student_checkpoint, map_location=self.device)
            require_action_reference(ckpt, env.cfg, 'Student checkpoint')
            require_collision_signature(ckpt, env.collision_hash, 'Student checkpoint')
            if ckpt['asset_sha256'] != env.asset_hash:
                raise ValueError('Student asset mismatch')
            self.student.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.normalizer.load_state_dict(ckpt['running_mean_std'])
            self.start_step = ckpt['step']

    def save(self, step):
        path = self.run_dir/'stage2_nn'
        path.mkdir(exist_ok=True)
        torch.save(dict(model=self.student.state_dict(), optimizer=self.optimizer.state_dict(),
                        running_mean_std=self.normalizer.state_dict(), network=self.spec,
                        history_dim=self.env.history_dim, history_length=self.env.prop_hist_len,
                        asset_sha256=self.env.asset_hash, collision_sha256=self.env.collision_hash,
                        step=step, kind='tactile_student',
                        joint_names=self.env.joint_names, lower=self.env.lower.cpu(), upper=self.env.upper.cpu(),
                        action_scale=self.env.cfg['action_scale'],
                        action_reference=action_reference(self.env.cfg),
                        action_smoothing=self.env.cfg['action_smoothing']), path/'last.pth')

    def train(self):
        obs = self.env.reset()
        writer = SummaryWriter(str(self.run_dir/'stage2_tb'))
        self.student.train()
        cfg = self.cfg.distill
        try:
            for step in range(self.start_step, cfg.steps):
                inputs = normalize_inputs(self.normalizer, obs['obs'], obs['proprio_hist'])
                with torch.no_grad():
                    target = self.teacher._actor_critic({'obs': inputs['obs'], 'priv_info': obs['priv_info']})
                prediction = self.student._actor_critic(inputs)
                loss, action_loss, latent_loss = distillation_loss(prediction, target, cfg.action_weight, cfg.latent_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite distillation loss')
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_norm)
                self.optimizer.step()
                # The student acts; teacher supervision is queried on those same visited states.
                with torch.no_grad():
                    action = self.student.act_inference(inputs).clamp(-1, 1)
                obs, reward, done, info = self.env.step(action)
                if step % 20 == 0:
                    for key, value in [('loss/total', loss), ('loss/action_nll', action_loss),
                                       ('loss/latent', latent_loss), ('reward', reward.mean())]:
                        writer.add_scalar(key, value.item(), step)
                    for key, value in info.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            writer.add_scalar(key, value.item(), step)
                if (step+1) % cfg.save_every == 0:
                    self.save(step+1)
                    print('distill step', step+1, 'loss', loss.item(), flush=True)
            self.save(cfg.steps)
        finally:
            writer.close()
