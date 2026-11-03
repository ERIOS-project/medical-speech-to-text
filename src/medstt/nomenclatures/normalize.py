"""Write a downloaded nomenclature as ``nomenclature/normalized/<name>.jsonl``."""

from __future__ import annotations

import logging
from pathlib import Path

from medstt.io import append_jsonl
from medstt.nomenclatures.config import NomenclatureSpec
from medstt.nomenclatures.fetch import fetch_source
from medstt.nomenclatures.importers import IMPORTERS

log = logging.getLogger(__name__)


def normalize(data_file: Path, spec: NomenclatureSpec, output: Path) -> int:
    """Convert one source file with its importer and return the number of concepts."""
    importer = IMPORTERS[spec.format]
    part = output.with_suffix(".jsonl.part")
    part.unlink(missing_ok=True)
    count = 0
    batch = []
    for concept in importer(data_file):
        batch.append(concept.as_dict())
        count += 1
        if len(batch) >= 1000:
            append_jsonl(part, batch)
            batch = []
    if batch:
        append_jsonl(part, batch)
    if count == 0:
        part.unlink(missing_ok=True)
        raise ValueError(f"nomenclatures.{spec.name}: no concept found in {data_file}.")
    part.replace(output)
    return count


def import_nomenclature(
    raw_root: Path, normalized_root: Path, spec: NomenclatureSpec, force: bool = False
) -> tuple[Path, int]:
    """Download (if needed) and normalize one nomenclature; return the output path and count."""
    data_file = fetch_source(raw_root, spec, force=force)
    output = normalized_root / f"{spec.name}.jsonl"
    count = normalize(data_file, spec, output)
    log.info("%s: %d concepts written to %s", spec.name, count, output)
    return output, count
