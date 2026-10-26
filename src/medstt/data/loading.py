"""Load AudioFolder or saved Hugging Face datasets for ASR training.

Every input is mapped to ``audio``, ``text``, ``word``, and ``source`` columns
before multiple datasets are combined. Audio stays undecoded until batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_from_disk

from medstt.io import read_records
from medstt.workspace import Workspace

COLUMNS = ("audio", "text", "word", "source")


@dataclass(frozen=True)
class DatasetSpec:
    """Location and optional column, split, sampling, and weighting settings."""

    path: str
    name: str | None = None
    split: str | None = None
    audio_column: str = "audio"
    text_column: str = "text"
    word_column: str = "word"
    max_samples: int | None = None
    #: Reuse each row this many times when combining datasets.
    repeat: int = 1

    @classmethod
    def parse(cls, value: str | dict[str, Any]) -> DatasetSpec:
        """Accept a path string or a mapping of dataset options."""
        if isinstance(value, str):
            return cls(path=value)
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown dataset options {sorted(unknown)} in {value}.")
        return cls(**value)


def load_dataset_spec(ws: Workspace, spec: DatasetSpec, seed: int = 42):
    """Load one dataset and map it to the common four-column schema."""
    path = ws.resolve(spec.path, base="datasets")
    name = spec.name or path.name
    metadata = _find_metadata(path)
    if metadata is not None:
        records = read_records(metadata)
        if records and spec.text_column not in records[0]:
            raise ValueError(f"{metadata}: missing text column {spec.text_column!r}.")
        ds = Dataset.from_dict(
            {
                "audio": [str(path / r["file_name"]) for r in records],
                "text": [str(r[spec.text_column] or "") for r in records],
                "word": [str(r.get(spec.word_column) or "") for r in records],
            }
        )
    elif (path / "dataset_dict.json").exists() or (path / "state.json").exists():
        ds = _load_from_disk(path, spec)
    else:
        raise FileNotFoundError(
            f"{path} is neither an AudioFolder (metadata.csv/jsonl) nor a save_to_disk dataset."
        )

    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda text: bool(text.strip()), input_columns="text")
    if spec.max_samples is not None and len(ds) > spec.max_samples:
        ds = ds.shuffle(seed=seed).select(range(spec.max_samples))
    ds = ds.add_column("source", [name] * len(ds))
    if spec.repeat > 1:
        ds = concatenate_datasets([ds] * spec.repeat)
    return ds


def load_datasets(ws: Workspace, specs: list[DatasetSpec], seed: int = 42):
    """Concatenate datasets in the order listed in ``specs``."""
    if not specs:
        raise ValueError("At least one dataset is required.")
    parts = [load_dataset_spec(ws, spec, seed) for spec in specs]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _find_metadata(path: Path) -> Path | None:
    """Find an AudioFolder metadata file, if the directory has one."""
    for filename in ("metadata.csv", "metadata.jsonl"):
        if (path / filename).is_file():
            return path / filename
    return None


def _load_from_disk(path: Path, spec: DatasetSpec):
    """Read a saved Hugging Face dataset and select its configured columns."""
    ds = load_from_disk(str(path))
    if isinstance(ds, DatasetDict):
        split = spec.split or ("train" if "train" in ds else None)
        if split is None:
            raise ValueError(f"{path} has splits {list(ds)}; set `split` in the dataset spec.")
        ds = ds[split]
    for column in (spec.audio_column, spec.text_column):
        if column not in ds.column_names:
            raise ValueError(f"{path}: missing column {column!r} (got {ds.column_names}).")
    ds = ds.rename_columns(
        {
            old: new
            for old, new in ((spec.audio_column, "audio"), (spec.text_column, "text"))
            if old != new
        }
    )
    if spec.word_column in ds.column_names and spec.word_column != "word":
        ds = ds.rename_column(spec.word_column, "word")
    if "word" not in ds.column_names:
        ds = ds.add_column("word", [""] * len(ds))
    return ds.select_columns(["audio", "text", "word"])
