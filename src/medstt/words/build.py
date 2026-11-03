"""Build the word list from normalized nomenclatures.

Each concept contributes its rare words, taken from its label and synonyms.
A word is kept once: the first concept that mentions it, in the order of the
configured sources, becomes its context (code, label, definition). It keeps the
capitals the labels give it away from their first word (« Salmonella »).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medstt.io import append_jsonl, iter_jsonl
from medstt.text import strip_accents
from medstt.words.rarity import RaritySettings, Token, WordScore, candidate_tokens, score_word

log = logging.getLogger(__name__)


@dataclass
class Casing:
    """How the labels write a word: a capitalized spelling, and whether a lowercase one exists.

    A word keeps its capitals (« Salmonella », « Crohn », « SAI ») when some label writes
    it with a capital letter away from the first position and no label writes it in
    lowercase; the first word of a label only carries sentence case, so it tells nothing.
    """

    cased: str | None = None
    lowercase_seen: bool = False

    def observe(self, token: Token) -> None:
        """Record one occurrence of the word."""
        if not token.capitalized:
            self.lowercase_seen = True
        elif not token.initial and self.cased is None:
            self.cased = token.text

    def spelling(self, word: str) -> str:
        """Return the spelling to keep for the lowercase ``word``."""
        return self.cased if self.cased and not self.lowercase_seen else word


@dataclass
class BuildStats:
    """Counts gathered while building a word list, for the ``--stats`` report."""

    concepts: Counter = field(default_factory=Counter)
    without_rare_word: Counter = field(default_factory=Counter)
    words: Counter = field(default_factory=Counter)
    duplicates: int = 0
    #: Score of every distinct candidate word, for the histogram and threshold hints.
    scores: dict[str, WordScore] = field(default_factory=dict)

    @property
    def candidate_scores(self) -> dict[str, float]:
        """Score used against the threshold, per distinct candidate word."""
        return {word: ws.score for word, ws in self.scores.items()}

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
        lemmatized = [ws for ws in self.scores.values() if ws.lemmatized]
        rescued = [ws for ws in lemmatized if ws.zipf <= max_zipf < ws.lemma_zipf]
        lines.append(
            f"candidate words changed by the lemmatizer: {len(lemmatized)} of "
            f"{len(self.scores)}; dropped because their lemma is frequent: {len(rescued)}"
        )
        sample = sorted(rescued, key=lambda ws: -ws.lemma_zipf)[:30]
        lines.append(
            "  e.g. " + ", ".join(f"{ws.word}→{ws.lemma} {ws.lemma_zipf:.2f}" for ws in sample)
        )
        return "\n".join(lines)


def build_words(
    sources: list[tuple[str, Path]],
    output: Path,
    settings: RaritySettings,
    limit: int | None = None,
) -> tuple[int, BuildStats]:
    """Write one JSONL row per rare word and return the row count and statistics."""
    stats = BuildStats()
    casing: dict[str, Casing] = {}
    rows: dict[str, dict[str, Any]] = {}
    for source, path in sources:
        concepts = iter_jsonl(path)
        for index, concept in enumerate(concepts):
            if limit is not None and index >= limit:
                break
            stats.concepts[source] += 1
            found = False
            for label, main_label in _labels(concept):
                for token in candidate_tokens(label, settings.min_length):
                    word = token.word
                    ws = stats.scores.get(word)
                    if ws is None:
                        ws = stats.scores[word] = score_word(word)
                    if ws.score > settings.max_zipf:
                        continue
                    found = True
                    key = strip_accents(word)
                    casing.setdefault(key, Casing()).observe(token)
                    if key in rows:
                        stats.duplicates += 1
                        continue
                    stats.words[source] += 1
                    rows[key] = _row(ws, label, main_label, concept)
            if not found:
                stats.without_rare_word[source] += 1
    for key, row in rows.items():
        row["word"] = casing[key].spelling(row["word"])
    output.unlink(missing_ok=True)
    append_jsonl(output, list(rows.values()))
    return len(rows), stats


def _labels(concept: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return the label then each synonym, paired with the main label for synonyms."""
    label = str(concept.get("label") or "")
    pairs: list[tuple[str, str | None]] = [(label, None)]
    pairs += [(str(s), label) for s in concept.get("synonyms") or [] if s]
    return pairs


def _row(
    ws: WordScore, label: str, main_label: str | None, concept: dict[str, Any]
) -> dict[str, Any]:
    """Build the output record for one rare word of one concept."""
    extra = dict(concept.get("extra") or {})
    if main_label is not None:
        extra["label"] = main_label
    return {
        "word": ws.word,
        "zipf": round(ws.score, 2),
        "lemma": ws.lemma,
        "source": concept.get("source"),
        "code": concept.get("code"),
        "label": label,
        "definition": concept.get("definition"),
        "category": concept.get("category"),
        "synonyms": concept.get("synonyms") or [],
        "extra": extra,
    }
