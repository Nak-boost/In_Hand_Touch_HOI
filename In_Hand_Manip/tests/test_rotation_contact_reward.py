"""Experiment B must not reward reverse rotation less or alter baseline task settings."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
spec=importlib.util.spec_from_file_location('rotation_math',
    Path(__file__).resolve().parents[1]/'hora/tasks/manipulation_math.py')
maths=importlib.util.module_from_spec(spec);spec.loader.exec_module(maths)
contact_conditioned_rotation=maths.contact_conditioned_rotation
rotation_contact_multipliers=maths.rotation_contact_multipliers


def test_positive_progress_uses_filtered_count_and_reverse_penalty_is_preserved():
    contacts = torch.arange(4)[None, :] < torch.arange(5)[:, None]
    positive, scale = contact_conditioned_rotation(torch.ones(5), contacts, (.1,.5,1.))
    torch.testing.assert_close(positive, torch.tensor([.1,.5,1.,1.,1.]))
    negative, _ = contact_conditioned_rotation(-torch.ones(5), contacts, (.1,.5,1.))
    torch.testing.assert_close(negative, -torch.ones(5))
    zero, _ = contact_conditioned_rotation(torch.zeros(5), contacts, (.1,.5,1.))
    assert not zero.any()
    progress=torch.linspace(-1,1,5)
    baseline, scale=contact_conditioned_rotation(progress, contacts)
    assert torch.equal(baseline,progress) and torch.equal(scale,torch.ones(5))


def test_raw_self_contact_cannot_unlock_rotation_reward():
    from hora.utils.object_contacts import gel_object_normal_load
    lookup=np.array([0,1,2,3,-1])
    dtype=[('body0',np.int32),('body1',np.int32),('lambda',np.float32)]
    # Large self-contact loads cannot make a weak object contact cross 0.1 N.
    frames=[[(0,1,50),(2,3,50),(0,5,.04)],
            [(0,1,50),(2,3,50),(0,5,.13)]]
    filtered=torch.from_numpy(np.stack([
        gel_object_normal_load(np.array(rows,dtype=dtype),5,lookup)>.1 for rows in frames]))
    reward,_=contact_conditioned_rotation(torch.ones(2),filtered,(.1,.5,1.))
    torch.testing.assert_close(reward,torch.tensor([.1,.5]))
    with pytest.raises(ValueError,match='object_contact_filter'):
        rotation_contact_multipliers({'rotation_contact_multipliers':[.1,.5,1.]})


@pytest.mark.parametrize('values', [[.1,.5],[-.1,.5,1.],[.1,1.1,1.],
                                   [.5,.1,1.],[.1,float('nan'),1.],3])
def test_bad_configuration_fails_before_creating_simulation(values):
    with pytest.raises(ValueError):
        rotation_contact_multipliers({'rotation_contact_multipliers':values,'object_contact_filter':True})


def test_experiment_b_has_only_one_configuration_difference():
    config_dir=Path(__file__).resolve().parents[1]/'configs'
    with initialize_config_dir(version_base='1.3',config_dir=str(config_dir)):
        base=OmegaConf.to_container(compose(config_name='manip_hora16384_sphere55'),resolve=False)
        b=OmegaConf.to_container(compose(config_name='manip_hora16384_sphere55_gel_rotation'),resolve=False)
        tabletop=OmegaConf.to_container(compose(config_name='manip_tabletop_inward90_hora16384_sphere55'),resolve=False)
    assert b['task']['env']['manipulation'].pop('rotation_contact_multipliers')==[.1,.5,1.]
    assert b==base
    assert base['task']['env']['manipulation']['position_weight']==2.0
    assert base['task']['env']['manipulation']['contact_weight']==.05
    assert rotation_contact_multipliers(base['task']['env']['manipulation']) is None
    assert rotation_contact_multipliers(tabletop['task']['env']['manipulation']) is None
