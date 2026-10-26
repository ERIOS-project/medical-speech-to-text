"""Download model files and copy external datasets into the workspace."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from medstt.io import write_csv

log = logging.getLogger(__name__)


def import_dataset(
    source: str,
    output_dir: Path,
    *,
    config: str | None,
    split: str,
    audio_column: str,
    text_column: str,
    max_samples: int | None,
    streaming: bool,
) -> int:
    """Copy a Hub or local dataset to AudioFolder without decoding its audio.

    Return the number of rows whose audio could be written.
    """
    ds = load_dataset(source, config, split=split, streaming=streaming)
    ds = ds.cast_column(audio_column, Audio(decode=False))
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, row in enumerate(tqdm(ds, desc=f"import {source}", unit="utt", total=max_samples)):
        if max_samples is not None and index >= max_samples:
            break
        audio = row[audio_column]
        suffix = Path(audio.get("path") or "").suffix or ".wav"
        file_name = f"audio/{index:08d}{suffix}"
        if audio.get("bytes"):
            (output_dir / file_name).write_bytes(audio["bytes"])
        elif audio.get("path") and Path(audio["path"]).is_file():
            shutil.copyfile(audio["path"], output_dir / file_name)
        else:
            log.warning("Row %d has no readable audio, skipped.", index)
            continue
        scalars = {
            key: value
            for key, value in row.items()
            if key not in (audio_column, text_column) and isinstance(value, (str, int, float, bool))
        }
        records.append({"file_name": file_name, "text": row[text_column], **scalars})

    write_csv(output_dir / "metadata.csv", records, first_columns=("file_name", "text"))
    return len(records)


def download_model(repo_id: str, revision: str | None, local_dir: Path | None) -> Path:
    """Download a model snapshot to the HF cache or the requested directory."""
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir) if local_dir else None,
    )
    return Path(path)
