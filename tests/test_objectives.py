import torch

from sentinelbench.objectives import dpo_loss, grpo_loss, pairwise_reward_loss, ppo_loss


def test_dpo_backpropagates_toward_preferred_completion() -> None:
    chosen = torch.tensor([0.1, -0.3], requires_grad=True)
    rejected = torch.tensor([0.2, -0.5], requires_grad=True)
    metrics = dpo_loss(chosen, rejected, torch.zeros(2), torch.zeros(2), beta=0.5)

    metrics.loss.backward()

    assert metrics.loss.item() > 0
    assert chosen.grad is not None and torch.all(chosen.grad < 0)
    assert rejected.grad is not None and torch.all(rejected.grad > 0)


def test_reward_and_grpo_losses_are_finite() -> None:
    reward_metrics = pairwise_reward_loss(torch.tensor([2.0]), torch.tensor([1.0]))
    policy = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    policy_metrics = grpo_loss(policy, torch.zeros_like(policy), torch.tensor([[0.0, 1.0, 2.0]]))

    (reward_metrics.loss + policy_metrics.loss).backward()

    assert torch.isfinite(reward_metrics.loss)
    assert torch.isfinite(policy_metrics.loss)
    assert policy.grad is not None


def test_ppo_clips_overlarge_positive_policy_updates() -> None:
    policy = torch.tensor([2.0]).log().requires_grad_()
    old = torch.zeros(1)
    advantage = torch.ones(1)

    metrics = ppo_loss(policy, old, advantage, clip_epsilon=0.2)

    assert torch.allclose(metrics.loss, torch.tensor(-1.2))