"""Build the word list from normalized nomenclatures.

Each concept contributes its rare words, taken from its label and synonyms.
A word is kept once: the first concept that mentions it, in the order of the
configured sources, becomes its context (code, label, definition).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medstt.io import append_jsonl, iter_jsonl
from medstt.text import strip_accents
from medstt.words.rarity import RaritySettings, candidate_words, zipf

log = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """Counts gathered while building a word list, for the ``--stats`` report."""

    concepts: Counter = field(default_factory=Counter)
    without_rare_word: Counter = field(default_factory=Counter)
    words: Counter = field(default_factory=Counter)
    duplicates: int = 0
    #: Zipf score of every distinct candidate word, for the histogram and threshold hints.
    candidate_scores: dict[str, float] = field(default_factory=dict)

    def report(self, max_zipf: float) -> str:
        """Format the counts as a short text report."""
        lines = ["concepts read / without rare word / words kept, per source:"]
        for source in self.concepts:
            lines.append(
                f"  {source}: {self.concepts[source]} / {self.without_rare_word[source]} / "
                f"{self.words[source]}"
            )
        lines.append(f"duplicate words skipped: {self.duplicates}")
        buckets = Counter(int(score) for score in self.candidate_scores.values())
        lines.append("distinct candidate words per Zipf band:")
        for band in range(0, 8):
            mark = " <= max_zipf" if band + 1 <= max_zipf else ""
            lines.append(f"  [{band}, {band + 1}): {buckets.get(band, 0)}{mark}")
        ordered = sorted(self.candidate_scores.items(), key=lambda item: item[1])
        below = [w for w, s in ordered if s <= max_zipf][-30:]
        above = [w for w, s in ordered if s > max_zipf][:30]
        lines.append(f"kept, closest to the threshold: {', '.join(below)}")
        lines.append(f"dropped, closest to the threshold: {', '.join(above)}")
        return "\n".join(lines)


def build_words(
    sources: list[tuple[str, Path]],
    output: Path,
    settings: RaritySettings,
    limit: int | None = None,
) -> tuple[int, BuildStats]:
    """Write one JSONL row per rare word and return the row count and statistics."""
    stats = BuildStats()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for source, path in sources:
        concepts = iter_jsonl(path)
        for index, concept in enumerate(concepts):
            if limit is not None and index >= limit:
                break
            stats.concepts[source] += 1
            found = False
            for label, main_label in _labels(concept):
                for word in candidate_words(label, settings.min_length):
                    score = stats.candidate_scores.setdefault(word, zipf(word))
                    if score > settings.max_zipf:
                        continue
                    found = True
                    key = strip_accents(word)
                    if key in seen:
                        stats.duplicates += 1
                        continue
                    seen.add(key)
                    stats.words[source] += 1
                    rows.append(_row(word, score, label, main_label, concept))
            if not found:
                stats.without_rare_word[source] += 1
    output.unlink(missing_ok=True)
    append_jsonl(output, rows)
    return len(rows), stats


def _labels(concept: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return the label then each synonym, paired with the main label for synonyms."""
    label = str(concept.get("label") or "")
    pairs: list[tuple[str, str | None]] = [(label, None)]
    pairs += [(str(s), label) for s in concept.get("synonyms") or [] if s]
    return pairs


def _row(
    word: str, score: float, label: str, main_label: str | None, concept: dict[str, Any]
) -> dict[str, Any]:
    """Build the output record for one rare word of one concept."""
    extra = dict(concept.get("extra") or {})
    if main_label is not None:
        extra["label"] = main_label
    return {
        "word": word,
        "zipf": round(score, 2),
        "source": concept.get("source"),
        "code": concept.get("code"),
        "label": label,
        "definition": concept.get("definition"),
        "category": concept.get("category"),
        "synonyms": concept.get("synonyms") or [],
        "extra": extra,
    }
