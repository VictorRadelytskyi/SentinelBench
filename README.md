# Security Post-Training Lab

Post-training lab for security agents with PyTorch, TRL, PEFT, and reproducible evaluation.

A compact but research-shaped project for post-training cybersecurity agents. It focuses on the actual stack cited in the role: PyTorch objectives, preference and rollout schemas, PEFT + TRL integration, DeepSpeed/FSDP launch configs, and vLLM serving guidance.

Portfolio highlight: a reproducible, offline-safe post-training benchmark for analyst-style decision making, with executable DPO, PPO, GRPO, and reward-model objectives plus a machine-readable research report.

## Capabilities

| Role requirement | Artifact |
| --- | --- |
| SFT training data | `SFTExample` with chat-message serialization and reference corpus helpers |
| Reward modeling / RLAIF | Bradley-Terry `pairwise_reward_loss` and rubric-carrying preference records |
| DPO | Differentiable reference-relative DPO loss plus `scripts/train_dpo.py` using TRL |
| PPO | Clipped surrogate objective over agent rollouts |
| GRPO | Group-normalized rewards without a learned value critic |
| Parameter-efficient tuning | QLoRA/LoRA model builder through PEFT |
| GPU scaling | ZeRO-3 and FSDP launch configurations |
| High-throughput inference | vLLM serving recipe |
| Research reporting | Runtime, package-version, data-inventory, and objective-sanity report |

## Why It Works On A CV

This project is designed to read well in a recruiter screen and still hold up under technical review:

1. It demonstrates the exact stack called out in the role: PyTorch, Transformers, PEFT, TRL, DPO, PPO, GRPO, reward modeling, and GPU-oriented deployment.
2. It is executable, not just descriptive: a reviewer can run the tests, inspect the report, and see objective sanity metrics and stack readiness immediately.
3. It shows research engineering discipline: typed data records, synthetic preference corpora, reproducible configs, and a clear separation between training, evaluation, and serving.

## Quickstart

The core algorithms require only PyTorch:

```powershell
python -m pytest
$env:PYTHONPATH = "src"; python -m sentinelbench.cli
```

Install the training stack to run a real TRL DPO job:

```powershell
python -m pip install -e ".[training,distributed]"
$env:PYTHONPATH = "src"; python scripts/train_dpo.py --model <base-model-id>
```

The CLI reports a JSON research snapshot: runtime, optional package versions, synthetic data inventory, and objective sanity metrics. It does not download a checkpoint or allocate a GPU.

If you want a one-line summary for a CV, use this:

> Built a reproducible post-training lab for security agents with PyTorch, TRL, PEFT, DPO/PPO/GRPO objectives, and GPU-oriented deployment configs.

## Post-training workflow

1. Curate analyst demonstrations as `SFTExample` records and fine-tune a base policy.
2. Produce rubric-based preference pairs using expert review or an explicitly versioned AI judge; train a reward model with the pairwise objective.
3. Train with DPO when static preference pairs are sufficient, or PPO/GRPO when tool-use rollouts and online rewards are available.
4. Hold out scenario families and measure task success, harmful-action rate, calibration, and confidence intervals before promotion.
5. Merge or load adapters, serve with vLLM, and monitor schema fidelity, latency, action authorization, and drift.
6. Preserve an experiment manifest and a machine-readable report for every run.

## Algorithm Notes

`dpo_loss` uses $-\log \sigma(\beta[(\log \pi_\theta(y_w)-\log \pi_\theta(y_l))-(\log \pi_{ref}(y_w)-\log \pi_{ref}(y_l))])$ so a policy prefers analyst-approved completions without fitting a separate reward model.

`ppo_loss` implements the clipped surrogate $\min(r_t A_t, \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)$. `grpo_loss` normalizes rewards within each completion group and applies the same clipped update, avoiding a value model while retaining relative credit assignment.

## Scale and serving

Use [configs/deepspeed_zero3.json](configs/deepspeed_zero3.json) for memory-efficient ZeRO-3 training or [configs/fsdp.yaml](configs/fsdp.yaml) with Accelerate/FSDP. See [configs/vllm_serve.md](configs/vllm_serve.md) for inference-time tensor parallelism.

The examples are synthetic and contain no credentials, live telemetry, or destructive tooling. Security decisions should remain behind scoped, authenticated tool gateways in production.