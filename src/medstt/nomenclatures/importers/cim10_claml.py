"""Import the French CIM-10 published by ATIH in ClaML (XML) format.

Only leaf diagnostic categories are kept: chapters, blocks, and categories
with subcategories are too generic to dictate. Cross-reference codes nested in
labels (``<Reference>``) are dropped so they never end up glued to the text.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

from medstt.nomenclatures.importers.base import (
    Concept,
    clean_text,
    element_text,
    unique_texts,
)

SOURCE = "cim10"


def iter_cim10_concepts(path: Path) -> Iterator[Concept]:
    """Yield one concept per leaf category, with inclusion terms as synonyms."""
    kinds, parents = _hierarchy(path)
    for _, element in ElementTree.iterparse(path, events=("end",)):
        if element.tag != "Class":
            continue
        code = clean_text(element.get("code"))
        is_leaf = element.get("kind") == "category" and element.find("./SubClass") is None
        preferred = _rubric_texts(element, "preferred") if is_leaf and code else []
        if preferred:
            label = preferred[0]
            definitions = _rubric_texts(element, "definition")
            yield Concept(
                code=code,
                source=SOURCE,
                label=label,
                synonyms=unique_texts(_rubric_texts(element, "inclusion"), excluding=label),
                definition=definitions[0] if definitions else None,
                category=code[0],
                parent=parents.get(code),
                extra={"chapter": code[0], "block": _ancestor(code, "block", kinds, parents)},
            )
        element.clear()


def _hierarchy(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Collect every class kind and parent code in a first streaming pass."""
    kinds: dict[str, str] = {}
    parents: dict[str, str] = {}
    for _, element in ElementTree.iterparse(path, events=("end",)):
        if element.tag == "Class":
            code = clean_text(element.get("code"))
            if code:
                kinds[code] = element.get("kind") or ""
                parent = element.find("./SuperClass")
                if parent is not None and clean_text(parent.get("code")):
                    parents[code] = clean_text(parent.get("code"))
            element.clear()
    return kinds, parents


def _ancestor(code: str, kind: str, kinds: dict[str, str], parents: dict[str, str]) -> str | None:
    """Walk up the parent chain until a class of the requested kind is found."""
    current = parents.get(code)
    while current is not None:
        if kinds.get(current) == kind:
            return current
        current = parents.get(current)
    return None


def _rubric_texts(element: ElementTree.Element, kind: str) -> list[str]:
    """Return the label texts of rubrics of one kind, without reference codes."""
    texts = []
    for rubric in element.findall("./Rubric"):
        if rubric.get("kind") != kind:
            continue
        text = element_text(rubric.find("./Label"), skip=("Reference",))
        if text:
            texts.append(text)
    return texts
