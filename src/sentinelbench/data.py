from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTExample:
    """Analyst-approved demonstration for supervised fine-tuning."""

    system: str
    user: str
    assistant: str

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]


def reference_sft_examples() -> list[SFTExample]:
    return [
        SFTExample(
            system="You are a SOC analyst. Prefer evidence over urgency.",
            user="An endpoint launched encoded PowerShell after a suspicious login. What do you do?",
            assistant=(
                "Validate the identity evidence, check the endpoint telemetry, and only then contain "
                "the host and disable the account if the observations corroborate compromise."
            ),
        ),
        SFTExample(
            system="You are a SOC analyst. Prefer evidence over urgency.",
            user="A signed maintenance script ran during an approved change window. What do you do?",
            assistant=(
                "Confirm the change ticket, record the supporting telemetry, and close the alert as "
                "approved activity without containment."
            ),
        ),
    ]


@dataclass(frozen=True)
class PreferenceExample:
    """Pairwise preference record for DPO or reward-model training."""

    prompt: str
    chosen: str
    rejected: str
    rubric: str

    def validate(self) -> None:
        if not all((self.prompt.strip(), self.chosen.strip(), self.rejected.strip(), self.rubric.strip())):
            raise ValueError("Preference records require prompt, chosen, rejected, and rubric text")
        if self.chosen == self.rejected:
            raise ValueError("Chosen and rejected completions must differ")


def reference_preference_examples() -> list[PreferenceExample]:
    examples = [
        PreferenceExample(
            prompt="Investigate an alert where the user reported MFA fatigue and the endpoint launched encoded PowerShell.",
            chosen=(
                "I will preserve the alert context, verify identity and endpoint telemetry, then contain "
                "the host and disable the identity if both signals corroborate compromise."
            ),
            rejected="Disable every identity in the organization immediately.",
            rubric="Prefer scoped, evidence-based containment with proportional actions.",
        ),
        PreferenceExample(
            prompt="A signed maintenance script ran during an approved change window. What should the agent do?",
            chosen=(
                "Confirm the change ticket and host context, record the evidence, and close as expected "
                "activity without containment."
            ),
            rejected="Contain the host because every PowerShell event is malicious.",
            rubric="Prefer calibrated abstention and auditable evidence over blanket actions.",
        ),
    ]
    for example in examples:
        example.validate()
    return examples


@dataclass(frozen=True)
class Rollout:
    """A scored agent trajectory used by PPO/GRPO post-training."""

    prompt: str
    completion: str
    reward: float
    tool_trace: tuple[str, ...]

    def validate(self) -> None:
        if not self.completion.strip():
            raise ValueError("Rollouts require a completion")
        if not self.tool_trace:
            raise ValueError("Rollouts require an auditable tool trace")


def reference_rollouts() -> list[Rollout]:
    rollouts = [
        Rollout(
            prompt="Alert with suspicious login and encoded PowerShell.",
            completion=(
                "Checked identity telemetry, checked endpoint telemetry, then contained the host and "
                "disabled the identity because the evidence aligned with compromise."
            ),
            reward=1.0,
            tool_trace=("lookup_alert", "lookup_identity", "lookup_host", "disable_identity", "contain_host"),
        ),
        Rollout(
            prompt="Approved maintenance activity during a change window.",
            completion="Recorded the evidence and abstained from containment because the activity was approved.",
            reward=0.8,
            tool_trace=("lookup_alert", "lookup_identity", "lookup_host"),
        ),
    ]
    for rollout in rollouts:
        rollout.validate()
    return rollouts