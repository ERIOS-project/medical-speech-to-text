"""Import the French Orphanet nomenclature pack (``ORPHAnomenclature_fr_*.xml``)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

from medstt.nomenclatures.importers.base import Concept, clean_text, unique_texts

SOURCE = "orphanet"


def iter_orphanet_concepts(path: Path) -> Iterator[Concept]:
    """Yield one concept per ``Disorder``, with its synonyms and definition."""
    for _, element in ElementTree.iterparse(path, events=("end",)):
        if element.tag != "Disorder":
            continue
        code = clean_text(element.findtext("OrphaCode"))
        label = clean_text(element.findtext("Name"))
        if code and label:
            yield Concept(
                code=code,
                source=SOURCE,
                label=label,
                synonyms=unique_texts(
                    (node.text for node in element.findall("./SynonymList/Synonym")),
                    excluding=label,
                ),
                definition=_definition(element) or None,
                category=clean_text(element.findtext("./DisorderType/Name")) or None,
                parent=None,
                extra={
                    "classification_level": clean_text(
                        element.findtext("./ClassificationLevel/Name")
                    ),
                    "status": clean_text(element.findtext("Totalstatus")),
                    "expert_link": clean_text(element.findtext("ExpertLink")),
                },
            )
        element.clear()


def _definition(disorder: ElementTree.Element) -> str:
    """Return the text of the summary section named « Définition », if any."""
    for section in disorder.findall(".//TextSection"):
        section_type = clean_text(section.findtext("./TextSectionType/Name"))
        if section_type.casefold() == "définition":
            return clean_text(section.findtext("Contents"))
    return ""
