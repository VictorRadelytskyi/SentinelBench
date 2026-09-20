from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch

from .data import reference_preference_examples, reference_rollouts, reference_sft_examples
from .integrations import stack_summary
from .objectives import dpo_loss, grpo_loss, pairwise_reward_loss, ppo_loss


@dataclass(frozen=True)
class MetricSnapshot:
    loss: float
    preference_accuracy: float
    margin: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _snapshot(metrics) -> MetricSnapshot:
    return MetricSnapshot(
        loss=float(metrics.loss.detach().cpu().item()),
        preference_accuracy=float(metrics.preference_accuracy.detach().cpu().item()),
        margin=float(metrics.margin.detach().cpu().item()),
    )


def objective_sanity_report() -> dict[str, dict[str, float]]:
    dpo_metrics = dpo_loss(
        torch.tensor([0.7, 0.6]),
        torch.tensor([0.1, 0.2]),
        torch.zeros(2),
        torch.zeros(2),
        beta=0.5,
    )
    reward_metrics = pairwise_reward_loss(torch.tensor([1.7, 1.2]), torch.tensor([0.2, 0.1]))
    ppo_metrics = ppo_loss(
        torch.tensor([0.3, 0.2]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.4, 0.8]),
    )
    grpo_metrics = grpo_loss(
        torch.tensor([[-0.2231, -0.1054, 0.1823]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 3.0]]),
    )
    return {
        "dpo": _snapshot(dpo_metrics).as_dict(),
        "reward_model": _snapshot(reward_metrics).as_dict(),
        "ppo": _snapshot(ppo_metrics).as_dict(),
        "grpo": _snapshot(grpo_metrics).as_dict(),
    }


def project_report() -> dict[str, object]:
    stack = stack_summary()
    return {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "stack": stack,
        "data_inventory": {
            "sft_examples": len(reference_sft_examples()),
            "preference_examples": len(reference_preference_examples()),
            "rollouts": len(reference_rollouts()),
        },
        "objective_sanity": objective_sanity_report(),
        "readiness": {
            "core_algorithms": True,
            "training_stack_detected": all(stack[package]["installed"] for package in ("transformers", "peft", "trl", "datasets")),
        },
    }