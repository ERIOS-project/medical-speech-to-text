"""Download a nomenclature into ``nomenclature/raw/<name>/`` and locate its data file."""

from __future__ import annotations

import logging
import shutil
import zipfile
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from medstt.nomenclatures.config import NomenclatureSpec

log = logging.getLogger(__name__)

#: File types an importer can read, in order of preference.
DATA_SUFFIXES = (".xml",)


def fetch_source(raw_root: Path, spec: NomenclatureSpec, force: bool = False) -> Path:
    """Download ``spec.url`` once, unpack archives, and return the file to import."""
    target_dir = raw_root / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / _basename(spec.url)
    if force or not archive.exists():
        log.info("Downloading %s", spec.url)
        _download(spec.url, archive)
    else:
        log.info("Using existing download %s", archive)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            if force or not all((target_dir / name).exists() for name in members):
                zf.extractall(target_dir)
        candidates = [target_dir / name for name in members]
    else:
        candidates = [archive]
    return select_data_file(candidates, spec)


def select_data_file(candidates: list[Path], spec: NomenclatureSpec) -> Path:
    """Pick the file an importer should read, or explain how to disambiguate."""
    if spec.file:
        matches = [path for path in candidates if fnmatch(path.name, spec.file)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"nomenclatures.{spec.name}: `file: {spec.file}` matches "
                f"{[path.name for path in matches] or 'nothing'} in the download; "
                "it must match exactly one file."
            )
        return matches[0]
    data_files = [path for path in candidates if path.suffix.lower() in DATA_SUFFIXES]
    if len(data_files) == 1:
        return data_files[0]
    if not data_files:
        raise FileNotFoundError(
            f"nomenclatures.{spec.name}: no {DATA_SUFFIXES} file in the download "
            f"({[path.name for path in candidates]})."
        )
    raise ValueError(
        f"nomenclatures.{spec.name}: several data files found "
        f"({[path.name for path in data_files]}); set `file:` to choose one."
    )


def _basename(url: str) -> str:
    """Return the file name part of a URL, or a fallback name."""
    name = Path(urlparse(url).path).name
    return name or "download"


def _download(url: str, destination: Path) -> None:
    """Stream a URL into ``destination`` through a temporary ``.part`` file."""
    part = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, part.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    part.replace(destination)
