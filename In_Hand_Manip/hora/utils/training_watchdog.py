"""Small, simulator-independent persistence helpers for monitored teacher training."""
import json
import math
import os
import tempfile
import time
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(data, path):
    """A failed/interrupted save must leave the previous checkpoint intact."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TrainingJournal:
    def __init__(self, path=None):
        self.path = path
        self.state = dict(pid=os.getpid(), agent_steps=0, epoch_num=0)

    def write(self, phase, **values):
        if self.path:
            self.state.update(values, phase=phase, time=time.time())
            atomic_json(self.path, self.state)


def checkpoint_info(path):
    """Validate trusted local checkpoints, including optimizer and normalizers."""
    import torch
    data = torch.load(path, map_location='cpu')
    required = {'model', 'optimizer', 'running_mean_std', 'value_mean_std',
                'agent_steps', 'epoch_num', 'last_lr', 'asset_sha256',
                'collision_sha256', 'action_reference', 'kind'}
    if not required <= data.keys() or not data['model'] or data['kind'] != 'privileged_teacher':
        raise ValueError('Incomplete teacher checkpoint')
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return math.isfinite(value) if isinstance(value, float) else True
    if not finite(data) or not isinstance(data['agent_steps'], int) or data['agent_steps'] < 0:
        raise ValueError('Invalid/nonfinite checkpoint')
    return dict(path=str(Path(path).resolve()), agent_steps=data['agent_steps'],
                epoch_num=data['epoch_num'], asset_sha256=data['asset_sha256'],
                collision_sha256=data['collision_sha256'], action_reference=data['action_reference'])


def newest_checkpoint(paths):
    valid, rejected = [], []
    for path in sorted(set(map(str, paths))):
        try:
            valid.append(checkpoint_info(path))
        except Exception as error:
            rejected.append(dict(path=path, error=str(error)))
    return max(valid, key=lambda x: x['agent_steps']) if valid else None, rejected
