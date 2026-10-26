"""Run Cohere training from workspace settings, with checkpoint resumption."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from medstt.train.cohere import build_trainer
from medstt.train.common import build_splits, save_final
from medstt.train.config import TrainConfig
from medstt.workspace import Workspace

log = logging.getLogger(__name__)


def train(ws: Workspace, resume: bool) -> Path:
    """Train the model and return the final model directory.

    Checkpoints, metrics, and a copy of the settings stay in ``runs/<run_name>``.
    """
    train_settings = ws.section("train")
    cfg = TrainConfig.from_dict(train_settings)
    set_seed(cfg.seed)
    run_dir = ws.subdir("runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = build_splits(ws, cfg)
    log.info(
        "Train samples: %d | eval samples: %s",
        len(train_ds),
        len(eval_ds) if eval_ds is not None else "none",
    )

    trainer, processor = build_trainer(cfg, train_ds, eval_ds, run_dir)

    if trainer.is_world_process_zero():
        (run_dir / "train_config.yaml").write_text(
            yaml.safe_dump(
                train_settings,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    checkpoint = get_last_checkpoint(str(run_dir)) if resume else None
    if resume and checkpoint is None:
        log.warning("--resume given but no checkpoint found in %s; starting fresh.", run_dir)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_metrics("train", result.metrics)
    if eval_ds is not None:
        trainer.save_metrics("eval", trainer.evaluate())

    final = save_final(trainer, processor, run_dir)
    log.info("Model saved to %s", final)
    return final
