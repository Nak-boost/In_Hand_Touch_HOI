"""Guard against silently reusing a bank after controller or domain changes."""
import copy
import pytest
import torch
from hora.utils.grasp_validation import validation_contract, require_validation_contract, duration_steps, withdrawal_fraction
from hora.utils.manip_control import tracking_action, joint_targets


def task():
    return dict(sim=dict(dt=1/120,substeps=1),env=dict(
        controller=dict(controlFrequencyInv=6,pgain=3.,dgain=.1),
        manipulation=dict(action_reference='actual_position',action_scale=.026,action_smoothing=.5),
        digit=dict(contact_threshold=.1),domain=dict(mass=[.03,.12],gain_scale=[.9,1.1])))


@pytest.mark.parametrize('path,value',[
    (('sim','dt'),1/240),
    (('env','controller','controlFrequencyInv'),3),
    (('env','controller','pgain'),4.),
    (('env','manipulation','action_scale'),.05),
    (('env','digit','contact_threshold'),.05),
    (('env','domain','mass'),[.1,.2]),
])
def test_rejects_incompatible_validated_cache(path,value):
    original=task();bank={'validation_contract':validation_contract(original)}
    require_validation_contract(bank,original)
    changed=copy.deepcopy(original);parent=changed
    for key in path[:-1]:parent=parent[key]
    parent[path[-1]]=value
    with pytest.raises(ValueError,match='validation mismatch'):require_validation_contract(bank,changed)


@pytest.mark.parametrize('dt',[.025,.05,.1])
def test_withdrawal_uses_policy_path_and_physical_time(dt):
    cfg=task()['env']['manipulation'];q=torch.tensor([[.5]]);previous=torch.zeros_like(q)
    targets=q.clone();hold=q.clone();trajectory=[]
    for step in range(duration_steps(4.,dt)):
        desired=hold-.06*withdrawal_fraction((step+1)*dt)
        raw=tracking_action(q,targets,desired,previous,cfg)
        assert raw.abs().max()<=1
        previous=.5*raw+.5*previous
        targets,_=joint_targets(q,targets,.026*previous,torch.tensor([0.]),torch.tensor([1.]),'actual_position')
        q=targets.clone()  # Ideal plant: isolate command semantics from contact dynamics.
        trajectory.append(float(q))
    assert min(trajectory)<.445
    assert abs(trajectory[-1]-.5)<1e-5
    assert duration_steps(30.,dt)*dt==pytest.approx(30.)
