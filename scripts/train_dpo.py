"""Run DPO post-training over schema-validated analyst preference pairs."""

from __future__ import annotations

import argparse

from pathlib import Path

from sentinelbench.integrations import build_lora_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face causal-LM checkpoint")
    parser.add_argument("--dataset", default="examples/analyst_preferences.jsonl")
    parser.add_argument("--output", default="artifacts/dpo-security-agent")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    try:
        from datasets import load_dataset
        from trl import DPOConfig, DPOTrainer
    except ImportError as error:  # pragma: no cover - exercised only in stripped environments
        raise RuntimeError(
            "Install the project with the training extra to run DPO training."
        ) from error

    model = build_lora_causal_lm(arguments.model)
    dataset_path = Path(arguments.dataset)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=arguments.output,
            num_train_epochs=arguments.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
            learning_rate=arguments.learning_rate,
            bf16=True,
            logging_steps=1,
            report_to="none",
            seed=arguments.seed,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(arguments.output)


if __name__ == "__main__":
    main()