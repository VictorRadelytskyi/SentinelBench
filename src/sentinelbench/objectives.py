from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class ObjectiveMetrics:
    loss: Tensor
    preference_accuracy: Tensor
    margin: Tensor


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
) -> ObjectiveMetrics:
    """Direct Preference Optimization on sequence log probabilities.

    Inputs are per-example completion log probabilities, normally obtained by
    masking prompt tokens before summing token log probabilities.
    """
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_log_ratio - reference_log_ratio)
    loss = -functional.logsigmoid(logits).mean()
    return ObjectiveMetrics(
        loss=loss,
        preference_accuracy=(logits > 0).float().mean(),
        margin=logits.mean(),
    )


def pairwise_reward_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> ObjectiveMetrics:
    """Bradley-Terry reward-model loss for analyst-approved preference pairs."""
    margins = chosen_rewards - rejected_rewards
    return ObjectiveMetrics(
        loss=-functional.logsigmoid(margins).mean(),
        preference_accuracy=(margins > 0).float().mean(),
        margin=margins.mean(),
    )


def ppo_loss(
    policy_logps: Tensor,
    old_logps: Tensor,
    advantages: Tensor,
    clip_epsilon: float = 0.2,
) -> ObjectiveMetrics:
    """Clipped PPO surrogate for token- or sequence-level agent rollouts."""
    if policy_logps.shape != old_logps.shape or policy_logps.shape != advantages.shape:
        raise ValueError("policy_logps, old_logps, and advantages must have identical shapes")
    ratios = torch.exp(policy_logps - old_logps)
    unclipped = ratios * advantages
    clipped = ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    objective = torch.minimum(unclipped, clipped)
    return ObjectiveMetrics(
        loss=-objective.mean(),
        preference_accuracy=(advantages * (policy_logps - old_logps) > 0).float().mean(),
        margin=objective.mean(),
    )


def grpo_loss(
    policy_logps: Tensor,
    old_logps: Tensor,
    rewards: Tensor,
    clip_epsilon: float = 0.2,
) -> ObjectiveMetrics:
    """Group Relative Policy Optimization over completions for one prompt.

    The last dimension is a group of sampled completions for the same analyst
    task. Group-normalized rewards remove the need for a learned critic.
    """
    if policy_logps.shape != old_logps.shape or policy_logps.shape != rewards.shape:
        raise ValueError("policy_logps, old_logps, and rewards must have identical shapes")
    advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / (
        rewards.std(dim=-1, keepdim=True, correction=0).clamp_min(1e-6)
    )
    return ppo_loss(policy_logps, old_logps, advantages, clip_epsilon)