"""Control/domain contract recorded by physically revalidated grasp banks."""
import copy
import math
from .manip_control import action_reference


def validation_contract(task):
    env, sim = task['env'], task['sim']
    cfg = env['manipulation']
    return dict(version=1, physics_dt=float(sim['dt']), substeps=sim['substeps'],
                controller=copy.deepcopy(env['controller']),
                action_reference=action_reference(cfg), action_scale=cfg['action_scale'],
                action_smoothing=cfg['action_smoothing'],
                contact_threshold=env['digit']['contact_threshold'],
                domain=copy.deepcopy(env['domain']))


def require_validation_contract(bank, task):
    saved = bank.get('validation_contract')
    if saved is not None and saved != validation_contract(task):
        raise ValueError('Grasp cache control/domain validation mismatch; revalidate for this task')


def duration_steps(seconds, dt):
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Duration and control dt must be positive and finite')
    return int(math.ceil(seconds / dt - 1e-6))


def withdrawal_fraction(t):
    """1s hold, .5s withdraw, .5s hold, .5s restore, then hold."""
    return 0.0 if t <= 1 else min((t-1)/.5, 1.) if t <= 2 else max(1-(t-2)/.5, 0.)
