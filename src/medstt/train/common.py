"""Shared dataset, training-option, and model-saving helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from medstt.data.loading import load_datasets
from medstt.train.config import TrainConfig
from medstt.workspace import Workspace

log = logging.getLogger(__name__)


def build_splits(ws: Workspace, cfg: TrainConfig):
    """Load training data and use explicit evaluation data or a seeded holdout."""
    train = load_datasets(ws, cfg.data.train, seed=cfg.seed)
    if cfg.data.eval:
        return train, load_datasets(ws, cfg.data.eval, seed=cfg.seed)
    if cfg.data.eval_split > 0:
        split = train.train_test_split(test_size=cfg.data.eval_split, seed=cfg.seed)
        return split["train"], split["test"]
    return train, None


def training_arguments(
    args_cls,
    cfg: TrainConfig,
    run_dir: Path,
    has_eval: bool,
    defaults: dict[str, Any],
):
    """Apply project defaults, model defaults, then ``train.training`` overrides.

    Evaluation is disabled when there is no evaluation dataset.
    """
    values: dict[str, Any] = {
        "output_dir": str(run_dir),
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "warmup_steps": 200,
        "lr_scheduler_type": "linear",
        "logging_steps": 25,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 3,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        # Enable bf16 by default when PyTorch reports a CUDA/ROCm device.
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        # Non-reentrant checkpointing supports this encoder-decoder model.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 4,
        # Keep raw audio columns for decoding inside the collator.
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "ddp_find_unused_parameters": False,
        "report_to": "none",
    }
    values.update(defaults)
    values.update(cfg.training)
    if not has_eval:
        values["eval_strategy"] = "no"
        values["load_best_model_at_end"] = False
    return args_cls(**values)


def log_trainable_parameters(model) -> None:
    """Log the number and percentage of model weights that will be updated."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / max(total, 1),
    )


def save_final(trainer, processor, run_dir: Path) -> Path:
    """Save full model weights and the processor in ``<run>/final``."""
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(final_dir))
    return final_dir
