"""Sensor-to-joint-target adapter for an exported tactile student; no hardware I/O.

The caller supplies synchronized q/qdot, calibrated binary DIGIT contacts, and a task-frame
axis, and sends the returned targets to a rate/torque-limited PD controller at 20 Hz.
"""
import torch
from hora.algo.models.tactile_student import TactileStudent
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import normalize_inputs
from hora.utils.manip_control import action_reference, joint_targets


class DigitPolicyRuntime:
    def __init__(self, checkpoint, device='cpu'):
        self.device = device
        ckpt = torch.load(checkpoint, map_location=device)
        if ckpt.get('kind') != 'tactile_student':
            raise ValueError('Only a tactile_student checkpoint is deployable')
        self.model = TactileStudent(ckpt['network'], ckpt['history_dim']).to(device).eval()
        self.model.load_state_dict(ckpt['model'])
        self.rms = RunningMeanStd((ckpt['history_dim'],)).to(device).eval()
        self.rms.load_state_dict(ckpt['running_mean_std'])
        self.joint_names = ckpt['joint_names']
        self.lower = ckpt['lower'].to(device)
        self.upper = ckpt['upper'].to(device)
        self.scale, self.alpha = ckpt['action_scale'], ckpt['action_smoothing']
        self.action_reference = action_reference(ckpt)
        self.history = torch.zeros(1, ckpt['history_length'], ckpt['history_dim'], device=device)
        self.targets = None
        self.previous = torch.zeros(1, 16, device=device)

    def reset(self):
        self.targets = None
        self.previous.zero_()
        self.history.zero_()

    @torch.no_grad()
    def step(self, q, qdot, contacts, axis):
        values = [torch.as_tensor(v, dtype=torch.float32, device=self.device).reshape(1, -1)
                  for v in (q, qdot, contacts, axis)]
        q, qdot, contacts, axis = values
        if [v.shape[1] for v in values] != [16, 16, 4, 3] or not all(torch.isfinite(v).all() for v in values):
            raise ValueError('Expected finite q[16], qdot[16], contacts[4], axis[3]')
        if (contacts < 0).any() or (contacts > 1).any() or axis.norm() < 1e-6:
            raise ValueError('Contacts must be in [0,1], axis must be nonzero')
        axis = axis/axis.norm(dim=-1, keepdim=True)
        first = self.targets is None
        if first:
            self.targets = q.clamp(self.lower, self.upper).clone()
        normalize = lambda x: (2*(x-self.lower)/(self.upper-self.lower)-1).clamp(-1.5, 1.5)
        frame = torch.cat((normalize(q), (qdot/5).clamp(-3, 3), normalize(self.targets),
                           self.previous, contacts, axis), -1)
        self.history = frame[:, None].expand_as(self.history).clone() if first else torch.cat((self.history[:, 1:], frame[:, None]), 1)
        raw = self.model.act_inference(normalize_inputs(self.rms, frame, self.history)).clamp(-1, 1)
        self.previous = self.alpha*raw+(1-self.alpha)*self.previous
        self.targets, _ = joint_targets(q, self.targets, self.scale*self.previous,
                                       self.lower, self.upper, self.action_reference)
        return self.targets[0].clone()
