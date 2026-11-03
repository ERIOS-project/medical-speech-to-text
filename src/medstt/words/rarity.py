"""Find the words of a label that are rare in general French.

Rarity comes from ``wordfreq``, whose Zipf scale is the base-10 logarithm of a
word's occurrences per billion words: about 7 for « de », 4 for « maladie »,
2 for « mucopolysaccharidose », and 0 for a word absent from the lexicon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import simplemma
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


@dataclass(frozen=True)
class Token:
    """A candidate word as written in a label, with its lowercase form."""

    text: str
    word: str
    #: First word of the label: its capital may only be sentence case.
    initial: bool

    @property
    def capitalized(self) -> bool:
        """Whether the label writes the word with at least one capital letter."""
        return self.text != self.word


def candidate_tokens(text: str, min_length: int = 4) -> list[Token]:
    """Return the words of ``text`` worth scoring, in order, without repeats.

    Hyphenated tokens are split into their parts: nomenclatures (Orphanet in
    particular) use hyphens to chain whole phrases, as in
    « dysplasie épiphysaire multiple-macrocéphalie-dysmorphie faciale ».
    """
    seen: set[str] = set()
    tokens: list[Token] = []
    for index, match in enumerate(_WORD_RE.finditer(text)):
        token = _ELISION_RE.sub("", match.group(0)).replace("’", "'")
        for part, piece in enumerate(token.split("-")):
            word = piece.lower()
            key = strip_accents(word)
            if (
                len(key.replace("'", "")) < min_length
                or _ROMAN_RE.match(key)
                or word in STOPWORDS
                or key in seen
            ):
                continue
            seen.add(key)
            tokens.append(Token(piece, word, index == 0 and part == 0))
    return tokens


def candidate_words(text: str, min_length: int = 4) -> list[str]:
    """Return the lowercased words of ``text`` worth scoring, in order, without repeats."""
    return [token.word for token in candidate_tokens(text, min_length)]


@dataclass(frozen=True)
class WordScore:
    """Frequency of a word and of its lemma; ``score`` is the higher of the two."""

    word: str
    zipf: float
    lemma: str
    lemma_zipf: float

    @property
    def score(self) -> float:
        """Frequency used against ``max_zipf``."""
        return max(self.zipf, self.lemma_zipf)

    @property
    def lemmatized(self) -> bool:
        """Whether the lemmatizer changed the word."""
        return self.lemma != self.word


def zipf(word: str) -> float:
    """Return the French Zipf frequency of a word (0 when unknown)."""
    return zipf_frequency(word, "fr")


def lemmatize(word: str) -> str:
    """Return the French lemma of a word with ``simplemma``, or the word itself when unknown.

    Inflected forms of ordinary words are rarer than their lemma in the lexicon
    (« précisées » 2.95 against « préciser » 4.36), so the lemma frequency is needed
    to recognize them as ordinary. Medical words unknown to the lemmatizer stay as
    they are and keep their own frequency.
    """
    return simplemma.lemmatize(word, lang="fr").lower()


def score_word(word: str) -> WordScore:
    """Score a word with the frequency of its form and of its lemma."""
    lemma = lemmatize(word)
    return WordScore(word, zipf(word), lemma, zipf(lemma) if lemma != word else zipf(word))


def rare_words(text: str, settings: RaritySettings) -> list[tuple[str, float]]:
    """Return the candidate words of ``text`` whose score is at most ``max_zipf``."""
    scored = (score_word(word) for word in candidate_words(text, settings.min_length))
    return [(ws.word, ws.score) for ws in scored if ws.score <= settings.max_zipf]
