"""AnyRotate-inspired history encoder and latent + Gaussian action distillation."""
import torch
from torch import nn
from .models import ActorCritic


class TactileHistoryEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.temporal = nn.Sequential(nn.Conv1d(32, 32, 9, 2), nn.ReLU(),
                                      nn.Conv1d(32, 32, 5), nn.ReLU(),
                                      nn.Conv1d(32, 32, 5), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.project = nn.Linear(32, latent_dim)

    def forward(self, history):
        if history.shape[1] < 25:
            raise ValueError('TCN requires at least 25 history frames')
        return self.project(self.temporal(self.channel(history).transpose(1, 2)).squeeze(-1))


class TactileStudent(ActorCritic):
    def __init__(self, network_config, history_dim):
        cfg = dict(network_config)
        cfg['proprio_adapt'] = False
        super().__init__(cfg)
        self.priv_info_stage2 = True
        self.adapt_tconv = TactileHistoryEncoder(history_dim, network_config['priv_mlp_units'][-1])
        # Teacher labels are supplied separately; the student's privileged encoder is never used.
        for module in (self.env_mlp, self.value):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _actor_critic(self, obs_dict):
        # Deliberately discard priv_info even if a caller passes the full environment dictionary.
        return super()._actor_critic({'obs': obs_dict['obs'], 'proprio_hist': obs_dict['proprio_hist']})


def distillation_loss(student, teacher, action_weight=1.0, latent_weight=1.0):
    smu, slogstd, _, sz, _ = student
    tmu, tlogstd, _, tz, _ = teacher
    tmu, tlogstd, tz = tmu.detach(), tlogstd.detach(), tz.detach()
    # Exact expectation of action NLL under the teacher Gaussian, up to a constant.
    slogstd = slogstd.clamp(-5, 2)
    tlogstd = tlogstd.clamp(-5, 2)
    nll = (slogstd + 0.5*((tmu-smu).square()+torch.exp(2*tlogstd))*torch.exp(-2*slogstd)).mean()
    latent = (sz-tz).square().mean()
    return action_weight*nll+latent_weight*latent, nll, latent
