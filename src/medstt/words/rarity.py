"""Find the words of a label that are rare in general French.

Rarity comes from ``wordfreq``, whose Zipf scale is the base-10 logarithm of a
word's occurrences per billion words: about 7 for « de », 4 for « maladie »,
2 for « mucopolysaccharidose », and 0 for a word absent from the lexicon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wordfreq import zipf_frequency

from medstt.text import strip_accents

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
_ELISION_RE = re.compile(r"^(?:[ldjmntsc]|qu|jusqu|lorsqu|puisqu)[’']", re.IGNORECASE)
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")

#: Frequent function words and generic qualifiers that must never become a target.
_STOPWORD_TEXT = (
    "alors après autre autres aucun aucune avec avant chez comme dans depuis dont elle elles "
    "entre être hors leur leurs lors mais même mêmes moins notre nous parmi pendant plus "
    "pour quand sans selon sauf sont sous tout toute toutes tous très vers vous ainsi aussi "
    "cette ceux celle celles ceci cela"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split(" "))


@dataclass(frozen=True)
class RaritySettings:
    """Threshold and length rule used to keep a word."""

    max_zipf: float = 3.0
    min_length: int = 4


def candidate_words(text: str, min_length: int = 4) -> list[str]:
    """Return the lowercased words of ``text`` worth scoring, in order, without repeats.

    Hyphenated tokens are split into their parts: nomenclatures (Orphanet in
    particular) use hyphens to chain whole phrases, as in
    « dysplasie épiphysaire multiple-macrocéphalie-dysmorphie faciale ».
    """
    seen: set[str] = set()
    words: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = _ELISION_RE.sub("", match.group(0)).lower().replace("’", "'")
        for word in token.split("-"):
            key = strip_accents(word)
            if (
                len(key.replace("'", "")) < min_length
                or _ROMAN_RE.match(key)
                or word in STOPWORDS
                or key in seen
            ):
                continue
            seen.add(key)
            words.append(word)
    return words


def zipf(word: str) -> float:
    """Return the French Zipf frequency of a word (0 when unknown)."""
    return zipf_frequency(word, "fr")


def rare_words(text: str, settings: RaritySettings) -> list[tuple[str, float]]:
    """Return the candidate words of ``text`` whose frequency is at most ``max_zipf``."""
    scored = ((word, zipf(word)) for word in candidate_words(text, settings.min_length))
    return [(word, score) for word, score in scored if score <= settings.max_zipf]
