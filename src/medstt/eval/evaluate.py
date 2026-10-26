"""Placeholder for standalone ASR evaluation.

Training can already report WER and CER through ``train.data.eval`` or
``train.data.eval_split``. Standalone evaluation is still TODO.
"""

from __future__ import annotations

from medstt.workspace import Workspace


def evaluate_model(ws: Workspace, model: str, datasets: list[str]) -> dict:
    """Raise until a standalone evaluation protocol is implemented."""
    raise NotImplementedError(
        "Standalone evaluation is not implemented yet (see TODO.md). "
        "Use the evaluation metrics reported during training in the meantime."
    )
