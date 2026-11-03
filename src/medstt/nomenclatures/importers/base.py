"""Common concept record and text helpers shared by every importer."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Concept:
    """One entry of a nomenclature in the format shared by all sources."""

    code: str
    source: str
    label: str
    synonyms: list[str] = field(default_factory=list)
    definition: str | None = None
    category: str | None = None
    parent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record written to the normalized file."""
        return {
            "code": self.code,
            "source": self.source,
            "label": self.label,
            "synonyms": self.synonyms,
            "definition": self.definition,
            "category": self.category,
            "parent": self.parent,
            "extra": self.extra,
        }


def clean_text(value: str | None) -> str:
    """Collapse whitespace and strip a source string."""
    return _SPACES_RE.sub(" ", value or "").strip()


def unique_texts(values: Iterable[str | None], excluding: str = "") -> list[str]:
    """Keep nonempty strings once, case-insensitively, preserving source order."""
    seen = {excluding.casefold()} if excluding else set()
    result: list[str] = []
    for value in values:
        text = clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def element_text(element: ElementTree.Element | None, skip: tuple[str, ...] = ()) -> str:
    """Join an element's text, ignoring the subtrees whose tag is in ``skip``.

    Nested fragments are separated by a space so list items do not run together.
    """
    if element is None:
        return ""
    parts: list[str] = []

    def walk(node: ElementTree.Element) -> None:
        """Collect text nodes depth-first, skipping unwanted subtrees."""
        if node.text:
            parts.append(node.text)
        for child in node:
            if child.tag not in skip:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return clean_text(" ".join(parts))
