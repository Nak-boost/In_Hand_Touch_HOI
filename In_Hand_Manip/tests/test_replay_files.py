from pathlib import Path

import pytest
import yaml

from hora.utils.replay_files import resolve_checkpoint, replay_transfer_files


def test_checkpoint_resolution_rejects_ambiguous_policy_types(tmp_path):
    for stage in ['stage1_nn', 'stage2_nn']:
        (tmp_path / stage).mkdir()
        (tmp_path / stage / 'last.pth').write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='Ambiguous'):
        resolve_checkpoint(tmp_path, 'last')
    assert resolve_checkpoint(tmp_path, 'stage2_nn/last.pth') == tmp_path / 'stage2_nn/last.pth'


def test_checkpoint_resolution_accepts_named_training_snapshots(tmp_path):
    (tmp_path / 'stage1_nn').mkdir()
    path = tmp_path / 'stage1_nn/ep_100.pth'
    path.write_bytes(b'checkpoint')
    assert resolve_checkpoint(tmp_path, 'ep_100.pth') == path
    assert resolve_checkpoint(tmp_path, str(path)) == path
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path, 'best')


def test_transfer_selects_matching_cache_and_only_requested_checkpoint(tmp_path):
    run = tmp_path / 'runs/manip/example'
    (run / 'stage1_nn').mkdir(parents=True)
    cache = tmp_path / 'In_Hand_Manip/cache/custom.pt'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'cached states')
    (run / 'config.yaml').write_text(yaml.safe_dump({'task': {'env': {'grasp': {'cache': 'cache/custom.pt'}}}}))
    for name in ['best', 'last']:
        (run / 'stage1_nn' / (name + '.pth')).write_bytes(b'checkpoint')
    (run / 'events.out.tfevents.123').write_bytes(b'log')
    assert replay_transfer_files(tmp_path, run, 'best') == [
        'In_Hand_Manip/cache/custom.pt', 'runs/manip/example/config.yaml',
        'runs/manip/example/stage1_nn/best.pth']
    cache.unlink()
    with pytest.raises(FileNotFoundError, match='Required replay file'):
        replay_transfer_files(tmp_path, run, 'best')


def test_transfer_rejects_cache_outside_project(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/example'
    run.mkdir(parents=True)
    outside = tmp_path / 'outside.pt'
    outside.write_bytes(b'cache')
    (run / 'config.yaml').write_text(yaml.safe_dump({'task': {'env': {'grasp': {'cache': str(outside)}}}}))
    (run / 'last.pth').write_bytes(b'checkpoint')
    with pytest.raises(ValueError):
        replay_transfer_files(project, run)
