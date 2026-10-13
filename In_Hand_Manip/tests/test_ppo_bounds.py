"""Regression checks for unbiased action-mean regularization."""
import pytest
import torch

from hora.algo.ppo.ppo import action_bounds_loss


@pytest.mark.parametrize('bound', [1.0, 1.1])
def test_in_range_means_have_zero_loss_and_gradient(bound):
    mu = torch.tensor([[-bound, -0.5, 0., 0.5, bound]], requires_grad=True)
    loss = action_bounds_loss(mu, bound)
    assert loss.shape == (1,)
    assert torch.equal(loss, torch.zeros_like(loss))
    loss.sum().backward()
    assert torch.equal(mu.grad, torch.zeros_like(mu))


@pytest.mark.parametrize('bound', [1.0, 1.1])
def test_out_of_range_gradient_points_back_to_nearest_boundary(bound):
    mu = torch.tensor([[bound + 0.5, -bound - 0.5], [0., 0.]], requires_grad=True)
    loss = action_bounds_loss(mu, bound)
    torch.testing.assert_close(loss, torch.tensor([0.5, 0.]))
    torch.testing.assert_close(loss, action_bounds_loss(-mu, bound))
    loss.sum().backward()
    torch.testing.assert_close(mu.grad, torch.tensor([[1., -1.], [0., 0.]]))
    updated = mu.detach() - 0.1 * mu.grad
    assert action_bounds_loss(updated, bound)[0] < loss[0]
