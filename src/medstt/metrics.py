"""Compute word and character error rates for ASR predictions."""

from __future__ import annotations

import jiwer

from medstt.text import normalize_for_scoring


def text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Return normalized WER/CER and raw WER, ignoring empty references."""
    pairs = [
        (p, r) for p, r in zip(predictions, references, strict=True) if normalize_for_scoring(r)
    ]
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan"), "wer_raw": float("nan")}
    preds, refs = zip(*pairs, strict=True)
    preds_norm = [normalize_for_scoring(p) for p in preds]
    refs_norm = [normalize_for_scoring(r) for r in refs]
    return {
        "wer": jiwer.wer(refs_norm, preds_norm),
        "cer": jiwer.cer(refs_norm, preds_norm),
        "wer_raw": jiwer.wer([r.strip() for r in refs], [p.strip() for p in preds]),
    }
