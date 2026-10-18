"""Reward-shaping contracts, independent of the simulator."""
import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'hora/tasks/manipulation_math.py'
spec=importlib.util.spec_from_file_location('reward_math',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_bounded_off_axis_cost_is_smooth_monotone_and_preserves_small_error():
    x=torch.tensor([0.,1e-6,.25,1.,9.,100.,1e6],requires_grad=True)
    c=m.bounded_quadratic_cost(x,.15,.5)
    assert c[0]==0 and (c[1:]>c[:-1]).all() and (c<=.5).all()
    torch.testing.assert_close(c[1],.15*x[1],rtol=1e-5,atol=1e-12)
    c.sum().backward();assert torch.isfinite(x.grad).all() and (x.grad>=0).all()
    torch.testing.assert_close(m.bounded_quadratic_cost(x.detach(),.15,0),.15*x.detach())
    assert not m.bounded_quadratic_cost(x.detach(),0,.5).any()

def test_only_blocked_outward_motion_costs_and_retraction_remains_free():
    q=torch.tensor([[1.,-1.],[1.,-1.],[.99,0.],[0.,0.],[1.,-1.]])
    delta=torch.tensor([[.02,-.02],[-.02,.02],[.02,.02],[.02,-.02],[.002,-.002]])
    applied=(q+delta).clamp(-1,1)-q
    c=m.blocked_target_command(delta,applied,.02)
    torch.testing.assert_close(c,torch.tensor([1.,0.,.25,0.,.1]),atol=6e-6,rtol=1e-5)
    assert c[1]==0 and c[3]==0
    assert c[4]<c[0]  # Not a lost/requested ratio that penalizes tiny pushes equally.
    assert not m.blocked_target_command(torch.zeros(2,16),torch.zeros(2,16),.026).any()

def test_invalid_scales_rejected():
    with pytest.raises(ValueError):m.blocked_target_command(torch.zeros(1,16),torch.zeros(1,16),0)
    with pytest.raises(ValueError):m.bounded_quadratic_cost(torch.zeros(1),.15,-1)

def test_overspeed_cap_preserves_disabled_tabletop_term():
    fast=torch.tensor([0.,.01,100.,100000.])
    cost=m.bounded_quadratic_cost(fast,.1,.25)
    assert (cost>=0).all() and (cost<=.25).all()
    assert not m.bounded_quadratic_cost(fast,0,.25).any()

def test_rollout_diagnostics_include_failure_between_first_and_last_steps():
    from types import SimpleNamespace
    from hora.algo.ppo.ppo import PPO
    noop=lambda *args: None
    obs={'obs':torch.zeros(1,1),'priv_info':torch.zeros(1,1)}
    sequence=iter([(0.,0.),(-200.,1.),(0.,0.)])
    def step(actions):
        penalty,blocked=next(sequence)
        return obs,torch.tensor([penalty]),torch.tensor([penalty<0]),{
            'reward/failure_penalty':torch.tensor(penalty),
            'reward/total':torch.tensor(penalty),
            'control/blocked_target_command':torch.tensor(blocked),
            'other':penalty}
    storage=SimpleNamespace(update_data=noop,computer_return=noop,prepare_training=noop,
                            data_dict={'returns':torch.zeros(1),'values':torch.zeros(1)})
    agent=SimpleNamespace(horizon_length=3,obs=obs,storage=storage,
        model_act=lambda obs: {k:torch.zeros(1,1) for k in ['actions','neglogpacs','values','mus','sigmas']},
        env=SimpleNamespace(step=step),value_bootstrap=False,current_rewards=torch.zeros(1,1),
        current_lengths=torch.zeros(1),episode_rewards=SimpleNamespace(update=noop),
        episode_lengths=SimpleNamespace(update=noop),agent_steps=0,batch_size=3,gamma=.99,tau=.95,
        normalize_value=False)
    PPO.play_steps(agent)
    assert agent.extra_info['reward/failure_penalty'].item()==pytest.approx(-200/3)
    assert agent.extra_info['reward/total'].item()==pytest.approx(-200/3)
    assert agent.extra_info['control/blocked_target_command'].item()==pytest.approx(1/3)
    assert agent.extra_info['other']==0.
