"""Download a nomenclature and keep only its data file as ``nomenclature/raw/<name>.<ext>``."""

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
    """Return ``raw/<name>.<ext>``, downloading it from ``spec.url`` when missing.

    A zip archive is read once and only the data file it contains is kept, under the
    nomenclature name; the archive itself is discarded.
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_file(raw_root, spec.name)
    if existing and not force:
        log.info("Using existing download %s", existing)
        return existing

    log.info("Downloading %s", spec.url)
    download = raw_root / f"{spec.name}.download.part"
    try:
        _download(spec.url, download)
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as zf:
                members = [Path(name) for name in zf.namelist() if not name.endswith("/")]
                member = select_data_file(members, spec)
                target = raw_root / f"{spec.name}{member.suffix.lower()}"
                _replace_existing(raw_root, spec.name)
                with zf.open(member.as_posix()) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        else:
            name = Path(_basename(spec.url))
            select_data_file([name], spec)
            target = raw_root / f"{spec.name}{name.suffix.lower() or DATA_SUFFIXES[0]}"
            _replace_existing(raw_root, spec.name)
            download.replace(target)
    finally:
        download.unlink(missing_ok=True)
    return target


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


def _existing_file(raw_root: Path, name: str) -> Path | None:
    """Return the data file already kept for ``name``, if any."""
    files = sorted(
        path
        for path in raw_root.glob(f"{name}.*")
        if path.is_file() and not path.name.endswith(".part")
    )
    return files[0] if files else None


def _replace_existing(raw_root: Path, name: str) -> None:
    """Remove a previously kept data file so that only the new one remains."""
    while (existing := _existing_file(raw_root, name)) is not None:
        existing.unlink()


def _basename(url: str) -> str:
    """Return the file name part of a URL, or a fallback name."""
    name = Path(urlparse(url).path).name
    return name or "download"


def _download(url: str, destination: Path) -> None:
    """Stream a URL into ``destination``."""
    with urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
