"""Contact identity, load aggregation and backend contracts without Isaac Gym."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from hora.utils.object_contacts import gel_object_normal_load, validate_object_contact_filter


def contacts(rows):
    return np.array(rows, dtype=[('body0', np.int32), ('body1', np.int32), ('lambda', np.float32)])


def lookup():
    result = np.full(25, -1, dtype=np.int64)
    result[[6, 18, 24, 12]] = np.arange(4)
    return result


def test_only_object_pair_loads_count_in_both_body_orders():
    data = contacts([(25, 6, .06), (6, 25, .07),  # one manifold, sum > threshold
                     (25, 18, .04), (18, 3, 50),  # self-force must not lift weak object contact
                     (24, 12, 10), (12, 26, 10), (12, -1, 10),  # gel/gel, table, ground
                     (25, 5, 20), (25, 24, 0)])  # housing, zero-load candidate
    loads = gel_object_normal_load(data, 25, lookup())
    np.testing.assert_allclose(loads, [.13, .04, 0, 0], atol=1e-7)
    assert (loads > .1).tolist() == [True, False, False, False]


def test_empty_or_self_only_frame_clears_all_object_contacts():
    for rows in ([], [(6, 12, 8), (18, 5, 9)], [(25, 6, -1)]):
        assert not gel_object_normal_load(contacts(rows), 25, lookup()).any()


def test_nonfinite_object_contact_is_not_silent_loss_of_contact():
    with pytest.raises(FloatingPointError):
        gel_object_normal_load(contacts([(25, 6, float('nan'))]), 25, lookup())


def test_filter_requires_explicit_supported_backend():
    validate_object_contact_filter(False, {'use_gpu_pipeline': True})
    validate_object_contact_filter(True, {'use_gpu_pipeline': False, 'physx': {'use_gpu': True}})
    with pytest.raises(ValueError, match='pipeline=cpu'):
        validate_object_contact_filter(True, {'use_gpu_pipeline': True})
    with pytest.raises(ValueError, match='contact_collection'):
        validate_object_contact_filter(True, {'use_gpu_pipeline': False, 'physx': {'contact_collection': 0}})
    with pytest.raises(ValueError, match='boolean'):
        validate_object_contact_filter('false', {})


def test_self_contact_cannot_reset_object_loss_timer():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('tabletop_math', root/'hora/tasks/tabletop_math.py')
    maths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maths)
    raw = torch.tensor([[True, True, False, False]])
    effective = torch.from_numpy(gel_object_normal_load(contacts([(6, 12, 8)]), 25, lookup()) > .1)[None]
    assert bool(raw.sum(-1) >= 2) and not bool(effective.sum(-1) >= 2)
    position, center, missing = torch.zeros(1, 3), torch.zeros(2), torch.tensor([39])
    raw_failed, raw_missing, _, _ = maths.tabletop_failures(position, raw, missing, center, .1, 40)
    failed, effective_missing, _, lost = maths.tabletop_failures(position, effective, missing, center, .1, 40)
    assert not raw_failed.item() and raw_missing.item() == 0
    assert failed.item() and lost.item() and effective_missing.item() == 40


@pytest.mark.parametrize('name', ['manip', 'manip_tabletop_box45',
    'manip_compact27_sphere55', 'manip_compact27_sphere65', 'manip_compact27_box45',
    'manip_compact27_box55', 'manip_compact27_capsule70', 'manip_compact27_capsule80'])
def test_training_configs_enable_pair_filter_with_supported_pipeline(name):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1]
    # Same resolvers as train_manip; avoid importing Isaac Gym in this CPU test.
    if not OmegaConf.has_resolver('eq'):
        OmegaConf.register_new_resolver('eq', lambda x,y: str(x).lower()==str(y).lower())
    if not OmegaConf.has_resolver('contains'):
        OmegaConf.register_new_resolver('contains', lambda x,y: x.lower() in y.lower())
    if not OmegaConf.has_resolver('resolve_default'):
        OmegaConf.register_new_resolver('resolve_default', lambda default,arg: default if arg=='' else arg)
    with initialize_config_dir(version_base='1.3', config_dir=str(root/'configs')):
        cfg = compose(config_name=name)
    assert cfg.task.env.manipulation.object_contact_filter is True
    assert cfg.task.sim.use_gpu_pipeline is False
    assert cfg.task.sim.physx.use_gpu is True
    assert cfg.sim_device == 'cuda:0' and cfg.rl_device == 'cuda:0'
    validate_object_contact_filter(True, OmegaConf.to_container(cfg.task.sim, resolve=True))
