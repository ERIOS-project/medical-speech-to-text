"""Decode audio, convert it to mono, and resample it for TTS and training."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels when needed and return a one-dimensional float32 array."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # soundfile stores channels in the second dimension.
        audio = audio.mean(axis=1)
    return np.squeeze(audio).astype(np.float32, copy=False)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio only when its current rate differs from the target rate."""
    if orig_sr == target_sr:
        return audio
    return soxr.resample(audio, orig_sr, target_sr).astype(np.float32, copy=False)


def decode_audio(item: Any, target_sr: int) -> np.ndarray:
    """Read a path or Hugging Face audio record as mono float32 at ``target_sr``."""
    if isinstance(item, (str, Path)):
        audio, sr = sf.read(str(item), dtype="float32")
    elif isinstance(item, dict):
        if item.get("array") is not None:
            audio, sr = np.asarray(item["array"], dtype=np.float32), int(item["sampling_rate"])
        elif item.get("bytes"):
            audio, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32")
        elif item.get("path"):
            audio, sr = sf.read(item["path"], dtype="float32")
        else:
            raise ValueError("Audio item has neither array, bytes nor path.")
    else:
        raise TypeError(f"Unsupported audio item type: {type(item).__name__}")
    return resample(to_mono(audio), sr, target_sr)


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale the largest sample to ``peak``; leave silent audio unchanged."""
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs == 0.0:
        return audio
    return (audio * (peak / max_abs)).astype(np.float32, copy=False)
