"""Simulator-independent conventions for the palm-up, two-axis task (SI, xyzw)."""
import torch

OBS_DIM = 71  # q, qdot, q_target, previous action, four contacts, commanded axis
HISTORY_DIM = OBS_DIM
PRIV_DIM = 23  # position, quaternion, linear/angular velocity, dimensions, mass, friction, COM, goal phase


def joint_pose_deviation(q, initial_q):
    """Per-environment squared joint displacement from the episode's initial grasp."""
    return (q - initial_q).square().sum(dim=-1)


def quat_mul(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


def rotation_increment(current, previous):
    conjugate = previous * previous.new_tensor([-1, -1, -1, 1])
    delta = quat_mul(current, conjugate)
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # q and -q describe the same orientation; always take the short arc.
    delta = torch.where(delta[..., 3:] < 0, -delta, delta)
    length = delta[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, delta[..., 3:].clamp_min(0))
    return delta[..., :3] * torch.where(length > 1e-6, angle / length.clamp_min(1e-8), 2.0)


def build_observation(q, qdot, targets, previous_action, contacts, axis, lower, upper):
    normalize = lambda x: (2 * (x - lower) / (upper - lower) - 1).clamp(-1.5, 1.5)
    return torch.cat((normalize(q), (qdot / 5).clamp(-3, 3), normalize(targets),
                      previous_action, contacts, axis), -1)


def append_history(history, frame, reset_mask):
    updated = torch.cat((history[:, 1:], frame[:, None]), 1)
    return torch.where(reset_mask[:, None, None], frame[:, None].expand_as(updated), updated)


def rotation_terms(omega, axis, speed):
    along = (omega * axis).sum(-1)
    off_axis = (omega - along[:, None] * axis).square().sum(-1)
    # A bounded directional reward avoids optimizing uncontrolled high-speed spinning.
    progress = along.clamp(-speed, speed) / speed
    overspeed = (along - speed).clamp_min(0).square()
    return progress, off_axis, overspeed, along
