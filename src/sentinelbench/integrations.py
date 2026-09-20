from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version


TRAINING_PACKAGES = ("transformers", "peft", "trl", "datasets")
DEPLOYMENT_PACKAGES = ("deepspeed", "accelerate", "vllm")


@dataclass(frozen=True)
class PackageStatus:
    installed: bool
    version: str | None

    def as_dict(self) -> dict[str, str | bool | None]:
        return {"installed": self.installed, "version": self.version}


def stack_report() -> dict[str, bool]:
    """Return installed-stack readiness without importing optional GPU packages."""
    packages = (*TRAINING_PACKAGES, *DEPLOYMENT_PACKAGES)
    return {package: importlib.util.find_spec(package) is not None for package in packages}


def stack_summary() -> dict[str, dict[str, str | bool | None]]:
    summary: dict[str, dict[str, str | bool | None]] = {}
    for package, installed in stack_report().items():
        version = None
        if installed:
            try:
                version = package_version(package)
            except PackageNotFoundError:
                version = None
        summary[package] = PackageStatus(installed=installed, version=version).as_dict()
    return summary


def missing_training_dependencies() -> list[str]:
    return [package for package in TRAINING_PACKAGES if not stack_report()[package]]


def build_lora_causal_lm(model_name: str):
    """Load a causal LM and attach QLoRA adapters when training extras are installed."""
    missing = missing_training_dependencies()
    if missing:
        raise RuntimeError(f"Install the training extra before use: missing {', '.join(missing)}")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    return get_peft_model(model, config)