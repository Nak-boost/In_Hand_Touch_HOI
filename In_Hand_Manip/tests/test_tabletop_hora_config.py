"""The 16384 runs use HORA optimizer settings with the tabletop task interface."""
from pathlib import Path
import pytest
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('obj',['sphere55','box45'])
def test_hora_training_parameters_and_tabletop_contract(obj):
    c=OmegaConf.load(ROOT/'configs'/('manip_tabletop_inward90_hora16384_'+obj+'.yaml'))
    reference=OmegaConf.load(ROOT/'configs/train/AllegroHandHora.yaml')
    assert c.num_envs==c.task.env.numEnvs==c.train.ppo.num_actors==16384
    assert c.task.name=='AllegroDigitTabletop'
    assert c.task.env.objects[0].name==obj
    assert c.task.env.asset.handAsset.endswith('compact_inward90.urdf')
    assert c.task.env.manipulation.action_reference=='actual_position'
    assert c.task.env.manipulation.object_contact_filter
    assert not c.task.sim.use_gpu_pipeline and c.task.sim.physx.use_gpu
    baseline=OmegaConf.load(ROOT/'configs'/('manip_tabletop_inward90_scale2560_'+obj+'.yaml'))
    assert c.train.network==baseline.train.network
    assert list(c.train.network.mlp.units)==[256,256,128]
    assert list(c.train.network.priv_mlp.units)==[128,64,8]
    for key in reference.ppo:
        if key not in ['num_actors','priv_info','priv_info_dim','value_bootstrap']:
            assert c.train.ppo[key]==reference.ppo[key],key
    assert c.train.ppo.value_bootstrap==baseline.train.ppo.value_bootstrap==False
    assert c.train.ppo.priv_info and c.train.ppo.priv_info_dim==23
    assert c.num_envs*c.train.ppo.horizon_length==131072
    assert (131072//c.train.ppo.minibatch_size)*c.train.ppo.mini_epochs==20


@pytest.mark.parametrize('obj', ['sphere55', 'box45'])
def test_only_authorized_training_fields_differ(obj):
    """Guard all task/network fields, including ones not named in the checks above."""
    baseline = OmegaConf.to_container(OmegaConf.load(
        ROOT/'configs'/('manip_tabletop_inward90_scale2560_'+obj+'.yaml')), resolve=True)
    candidate = OmegaConf.to_container(OmegaConf.load(
        ROOT/'configs'/('manip_tabletop_inward90_hora16384_'+obj+'.yaml')), resolve=True)
    allowed = {
        'num_envs', 'task.env.numEnvs', 'train.ppo.num_actors',
        'train.ppo.horizon_length', 'train.ppo.minibatch_size',
        'train.ppo.mini_epochs', 'train.ppo.learning_rate',
        'train.ppo.save_frequency', 'train.ppo.max_agent_steps',
    }

    def check(before, after, path=''):
        if path in allowed:
            return
        if isinstance(before, dict):
            assert isinstance(after, dict) and before.keys() == after.keys(), path
            for key in before:
                check(before[key], after[key], path+'.'+key if path else key)
        else:
            assert before == after, path

    check(baseline, candidate)
