"""
LoRA fine-tuning wrapper used to train generation_{k+1} on data produced by
generation_k.

This is a thin, dependency-light wrapper around HF `transformers` + `peft`.
It expects you to supply your own base checkpoint (local path or hub id) and
compute; nothing here downloads or bundles model weights.

If you don't have GPU access for a given experiment, use
`generation_pipeline.py --transfer-mode in_context` instead, which skips
fine-tuning entirely and just conditions the successor via prompting, as
suggested in the project brief as a way to keep the setup tractable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FinetuneConfig:
    base_model: str                 # local path or HF hub id
    output_dir: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    max_seq_length: int = 1024


def _format_example(example: dict) -> str:
    """Turn a {prompt, completion} record into a single training string."""
    return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"


def write_dataset_jsonl(examples: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps({"text": _format_example(ex)}) + "\n")
    return out_path


def run_finetune(examples: list[dict], config: FinetuneConfig) -> str:
    """
    Fine-tune `config.base_model` on `examples` using LoRA and return the
    path to the resulting adapter.

    Requires: transformers, peft, datasets, accelerate, torch.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
    except ImportError as e:
        raise ImportError(
            "run_finetune requires torch, transformers, peft, datasets, accelerate. "
            "Install them or use --transfer-mode in_context to skip fine-tuning."
        ) from e

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    dataset_path = write_dataset_jsonl(examples, Path(config.output_dir) / "train.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    raw_dataset = load_dataset("json", data_files=str(dataset_path))["train"]

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    adapter_path = str(Path(config.output_dir) / "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    return adapter_path
