"""Normalize French text for term matching and transcription metrics."""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*")
_SPACES_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Remove accent marks while keeping the underlying letters."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_for_scoring(text: str) -> str:
    """Normalize case, punctuation, and spacing while preserving accents."""
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return _SPACES_RE.sub(" ", text).strip()


def normalize_for_matching(text: str) -> str:
    """Return a form suitable for accent- and case-insensitive matching."""
    return normalize_for_scoring(strip_accents(text))


def contains_term(text: str, term: str) -> bool:
    """Check for a complete term after normalizing accents and case."""
    term_norm = normalize_for_matching(term)
    if not term_norm:
        return False
    return f" {term_norm} " in f" {normalize_for_matching(text)} "


def count_words(text: str) -> int:
    """Count words, keeping hyphenated and apostrophe-linked forms together."""
    return len(_WORD_RE.findall(text))
