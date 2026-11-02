from types import SimpleNamespace

import pytest

from medstt.io import iter_jsonl
from medstt.metrics import text_metrics
from medstt.sentences.generate import (
    GenerationSettings,
    build_messages,
    clean_sentence,
    generate_sentences,
    validate_sentence,
)
from medstt.text import contains_term, count_words, normalize_for_scoring


def test_normalize_for_scoring_keeps_accents_and_drops_punctuation():
    assert normalize_for_scoring("  L’Échographie, normale ! ") == "l échographie normale"


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    [
        ("Une péricardite aiguë.", "pericardite", True),
        ("Une PÉRICARDITE aiguë.", "péricardite", True),
        ("Une péricardites aiguë.", "péricardite", False),
        ("Bilan de broncho-pneumopathie.", "broncho-pneumopathie", True),
        ("Texte", "", False),
    ],
)
def test_contains_term(text, term, expected):
    assert contains_term(text, term) is expected


def test_count_words_treats_hyphenated_and_elided_words_as_one():
    assert count_words("L'hôpital a vu une broncho-pneumopathie.") == 5


def test_clean_sentence_strips_reasoning_labels_and_quotes():
    raw = "<think>\nreasoning\n</think>\nSentence: « Le patient est stable. »\nExtra line"
    assert clean_sentence(raw) == "Le patient est stable."


def test_validate_sentence():
    sentence = "Le patient présente une tachycardie sinusale depuis ce matin."
    assert validate_sentence(sentence, "tachycardie", 5, 20) is None
    assert validate_sentence(sentence, "dyspnée", 5, 20) == "term missing"
    assert validate_sentence(sentence, "tachycardie", 12, 20).startswith("9 words")
    assert validate_sentence("", "tachycardie", 5, 20) == "empty reply"


def test_text_metrics_ignores_case_and_punctuation_for_wer():
    metrics = text_metrics(["le patient, va bien"], ["Le patient va bien."])
    assert metrics["wer"] == 0.0
    assert metrics["wer_raw"] > 0.0


def test_per_word_generation_resumes_without_duplicate_ids(tmp_path, monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(
                content="Le patient présente une tachycardie sinusale depuis ce matin."
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        async def close(self):
            pass

    monkeypatch.setattr("medstt.sentences.generate.AsyncOpenAI", FakeLLM)
    settings = GenerationSettings(
        base_url="http://unused",
        api_key="unused",
        model="fake-llm",
        domain="pratique clinique",
        min_words=5,
        max_words=20,
        temperature=0.8,
        max_tokens=100,
        concurrency=2,
        timeout=5,
        max_attempts=1,
        per_word=2,
    )
    output = tmp_path / "sentences.jsonl"
    terms = [{"word": "tachycardie"}]

    assert generate_sentences(terms, "word", output, settings) == (2, 0)
    assert generate_sentences(terms, "word", output, settings) == (0, 0)
    assert len(calls) == 2
    assert {row["id"] for row in iter_jsonl(output)} == {"000000-0", "000000-1"}


def _settings(**overrides) -> GenerationSettings:
    base = {
        "base_url": "http://unused",
        "api_key": "unused",
        "model": "fake-llm",
        "domain": "pratique clinique",
        "min_words": 8,
        "max_words": 20,
        "temperature": 0.8,
        "max_tokens": 100,
        "concurrency": 1,
        "timeout": 5,
        "max_attempts": 1,
        "per_word": 1,
    }
    return GenerationSettings(**{**base, **overrides})


def test_build_messages_adds_label_and_definition_when_present():
    record = {
        "word": "varioliforme",
        "label": "Acné varioliforme",
        "definition": "Forme d'acné laissant des cicatrices.",
        "extra": {"chapter": "L"},
    }
    user = build_messages("varioliforme", record, _settings())[1]["content"]

    assert "« varioliforme » écrit exactement ainsi" in user
    assert "Ce mot vient du libellé « Acné varioliforme »." in user
    assert "Définition du concept : Forme d'acné laissant des cicatrices." in user
    assert "entre 8 et 20 mots" in user
    assert "\n\n" not in user


def test_build_messages_without_context_for_a_bare_word_list():
    user = build_messages("tachycardie", {"word": "tachycardie"}, _settings())[1]["content"]
    bare = build_messages("x", {"word": "x", "label": "X", "definition": None}, _settings())

    assert "libellé «" not in user and "Définition" not in user
    assert "\n\n" not in user and not user.endswith("\n")
    assert "libellé «" not in bare[1]["content"]


def test_build_messages_accepts_custom_placeholders():
    settings = _settings(user_prompt="{word} / {label} / {definition} / {code} / {context}")
    record = {"word": "a", "label": "A b", "definition": "", "code": "A00"}

    assert build_messages("a", record, settings)[1]["content"] == (
        "a / A b /  / A00 / Ce mot vient du libellé « A b »."
    )
