"""Read input records and write JSONL or CSV pipeline outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

TABLE_SUFFIXES = (".txt", ".csv", ".tsv", ".jsonl", ".json", ".parquet")


def read_records(path: Path, txt_column: str = "text") -> list[dict[str, Any]]:
    """Read records from a supported file; each nonempty TXT line becomes one row."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [{txt_column: line.strip()} for line in lines if line.strip()]
    if suffix in (".csv", ".tsv"):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix == ".jsonl":
        return list(iter_jsonl(path))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list of objects.")
        return data
    if suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported file type {suffix!r} for {path}; expected {TABLE_SUFFIXES}.")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows, or no rows when the file does not exist."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append records as UTF-8 JSONL, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_csv(path: Path, records: list[dict[str, Any]], first_columns: tuple[str, ...]) -> None:
    """Write rows to CSV with selected columns first and remaining keys afterward."""
    columns = list(first_columns)
    for record in records:
        columns += [key for key in record if key not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def pick_column(record: dict[str, Any], candidates: tuple[str, ...], path: Path) -> str:
    """Return the first available column name, or explain what is missing."""
    for column in candidates:
        if column in record:
            return column
    raise ValueError(f"{path}: none of the columns {candidates} found (got {list(record)}).")
