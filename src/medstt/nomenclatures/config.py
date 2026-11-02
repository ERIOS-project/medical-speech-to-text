"""Parse the ``nomenclatures`` section of ``medstt.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medstt.nomenclatures.importers import IMPORTERS

_KNOWN_KEYS = {"format", "url", "file"}


@dataclass(frozen=True)
class NomenclatureSpec:
    """One source nomenclature: its importer format, download URL, and archive member."""

    name: str
    format: str
    url: str
    #: File to import when the downloaded archive holds several candidates.
    file: str | None = None

    @classmethod
    def parse(cls, name: str, raw: Any) -> NomenclatureSpec:
        """Validate one ``nomenclatures.<name>`` mapping."""
        if not isinstance(raw, dict):
            raise ValueError(f"nomenclatures.{name} must be a mapping with `format` and `url`.")
        unknown = set(raw) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"nomenclatures.{name}: unknown keys {sorted(unknown)}.")
        fmt, url = raw.get("format"), raw.get("url")
        if not fmt or not url:
            raise ValueError(f"nomenclatures.{name} requires `format` and `url`.")
        if fmt not in IMPORTERS:
            raise ValueError(
                f"nomenclatures.{name}: unknown format {fmt!r}; expected one of "
                f"{sorted(IMPORTERS)}."
            )
        return cls(name=name, format=fmt, url=str(url), file=raw.get("file") or None)


def parse_nomenclatures(section: dict[str, Any]) -> list[NomenclatureSpec]:
    """Return the configured nomenclatures in file order."""
    return [NomenclatureSpec.parse(name, raw) for name, raw in section.items()]
