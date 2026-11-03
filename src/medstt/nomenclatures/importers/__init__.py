"""Registry of nomenclature importers, keyed by the ``format`` setting."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from medstt.nomenclatures.importers.base import Concept
from medstt.nomenclatures.importers.cim10_claml import iter_cim10_concepts
from medstt.nomenclatures.importers.orphanet_xml import iter_orphanet_concepts

IMPORTERS: dict[str, Callable[[Path], Iterator[Concept]]] = {
    "cim10-claml": iter_cim10_concepts,
    "orphanet-xml": iter_orphanet_concepts,
}

__all__ = ["IMPORTERS", "Concept"]
