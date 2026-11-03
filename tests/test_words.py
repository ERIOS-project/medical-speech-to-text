"""Rare-word selection from normalized nomenclatures, with a fake frequency lexicon."""

import json

import pytest
from typer.testing import CliRunner

from medstt.cli import app
from medstt.io import append_jsonl
from medstt.words import build, rarity
from medstt.words.rarity import RaritySettings, candidate_tokens, candidate_words, rare_words
from medstt.workspace import Workspace

FREQUENCIES = {
    "maladie": 4.5,
    "infectieuse": 3.4,
    "choléra": 2.8,
    "cholera": 2.8,
    "classique": 4.1,
    "péricardite": 1.9,
    "varioliforme": 0.0,
    "acné": 3.2,
    "index": 4.0,
    "rare": 4.2,
}


@pytest.fixture(autouse=True)
def fake_lexicon(monkeypatch):
    monkeypatch.setattr(rarity, "zipf_frequency", lambda word, lang: FREQUENCIES.get(word, 0.0))


def test_candidate_words_normalizes_and_filters():
    words = candidate_words(
        "L'index et l’Acné, type IV, chez 2 enfants : Al-Gazali sans ADN, bras court-surdité"
    )
    assert words == ["index", "acné", "type", "enfants", "gazali", "bras", "court", "surdité"]


def test_rare_words_uses_threshold_and_min_length():
    found = rare_words("Acné varioliforme de la maladie", RaritySettings(max_zipf=3.0))
    assert found == [("varioliforme", 0.0)]
    assert rare_words("Acné varioliforme", RaritySettings(max_zipf=3.5)) == [
        ("acné", 3.2),
        ("varioliforme", 0.0),
    ]


def _normalized(path, rows):
    append_jsonl(path, rows)
    return path


def test_build_keeps_each_word_once_with_its_first_concept(tmp_path):
    cim10 = _normalized(
        tmp_path / "cim10.jsonl",
        [
            {
                "code": "A00.0",
                "source": "cim10",
                "label": "Choléra classique",
                "synonyms": ["Maladie infectieuse à Péricardite"],
                "definition": "Def A",
                "category": "A",
                "parent": "A00",
                "extra": {"chapter": "A"},
            },
            {
                "code": "L70.2",
                "source": "cim10",
                "label": "Acné varioliforme",
                "synonyms": [],
                "definition": None,
                "category": "L",
                "parent": None,
                "extra": {},
            },
            {
                "code": "Z00",
                "source": "cim10",
                "label": "Maladie classique",
                "synonyms": [],
                "definition": None,
                "category": "Z",
                "parent": None,
                "extra": {},
            },
        ],
    )
    orpha = _normalized(
        tmp_path / "orphanet.jsonl",
        [
            {
                "code": "1",
                "source": "orphanet",
                "label": "CHOLERA rare",
                "synonyms": [],
                "definition": "Def O",
                "category": "Maladie",
                "parent": None,
                "extra": {},
            }
        ],
    )
    output = tmp_path / "medical.jsonl"

    count, stats = build.build_words(
        [("cim10", cim10), ("orphanet", orpha)], output, RaritySettings(max_zipf=3.0)
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 3
    # « Péricardite » keeps its capital: the fixture only writes it capitalized, away from
    # the first word of the synonym.
    assert [row["word"] for row in rows] == ["choléra", "Péricardite", "varioliforme"]
    assert rows[0]["code"] == "A00.0" and rows[0]["definition"] == "Def A"
    assert rows[1]["label"] == "Maladie infectieuse à Péricardite"
    assert rows[1]["extra"] == {"chapter": "A", "label": "Choléra classique"}
    assert rows[2]["zipf"] == 0.0
    assert stats.concepts["cim10"] == 3 and stats.without_rare_word["cim10"] == 1
    assert stats.words == {"cim10": 3}
    assert stats.duplicates == 1  # "CHOLERA" from Orphanet matches "choléra"
    report = stats.report(3.0)
    assert "cim10: 3 / 1 / 3" in report and "orphanet: 1 / 0 / 0" in report
    assert "varioliforme" in report and "classique" in report


def test_build_limit_reads_the_first_concepts_only(tmp_path):
    rows = [
        {"code": str(i), "source": "cim10", "label": f"Choléra {i}", "synonyms": []}
        for i in range(5)
    ]
    source = _normalized(tmp_path / "cim10.jsonl", rows)

    count, stats = build.build_words(
        [("cim10", source)], tmp_path / "out.jsonl", RaritySettings(), limit=2
    )

    assert count == 1 and stats.concepts["cim10"] == 2


def test_cli_build_uses_yaml_sources_and_output(tmp_path):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "medstt.yaml").write_text(
        "nomenclatures:\n"
        "  cim10: {format: cim10-claml, url: https://example.org/c.xml}\n"
        "words: {sources: [cim10], output: med, max_zipf: 3.0, min_length: 4}\n",
        encoding="utf-8",
    )
    _normalized(
        ws.root / "nomenclature" / "normalized" / "cim10.jsonl",
        [{"code": "L70.2", "source": "cim10", "label": "Acné varioliforme", "synonyms": []}],
    )

    result = CliRunner().invoke(app, ["words", "build", "--stats", "--workdir", str(ws.root)])

    assert result.exit_code == 0, result.output
    assert "1 words written to" in result.output and "Zipf band" in result.output
    written = (ws.root / "words" / "med.jsonl").read_text(encoding="utf-8")
    assert json.loads(written)["word"] == "varioliforme"

    missing = CliRunner().invoke(app, ["words", "build", "orphanet", "--workdir", str(ws.root)])
    assert missing.exit_code != 0 and "nomenclature import orphanet" in missing.output


def test_score_uses_the_lemma_frequency_when_higher(monkeypatch):
    lexicon = {"précisées": 2.95, "préciser": 4.36, "infections": 4.5, "shigelloses": 0.0}
    monkeypatch.setattr(rarity, "zipf_frequency", lambda word, lang: lexicon.get(word, 0.0))
    monkeypatch.setattr(rarity, "lemmatize", lambda word: {"précisées": "préciser"}.get(word, word))

    precisees = rarity.score_word("précisées")
    assert (precisees.lemma, precisees.score, precisees.lemmatized) == ("préciser", 4.36, True)
    unknown = rarity.score_word("shigelloses")
    assert (unknown.lemma, unknown.score, unknown.lemmatized) == ("shigelloses", 0.0, False)
    assert rare_words("Autres infections précisées, shigelloses", RaritySettings(3.0)) == [
        ("shigelloses", 0.0)
    ]


def test_real_lemmatizer_keeps_unknown_medical_words():
    assert rarity.lemmatize("varioliforme") == "varioliforme"
    assert rarity.lemmatize("précisées") == "préciser"


def test_candidate_tokens_keep_the_label_spelling_and_position():
    tokens = candidate_tokens("Sepsis à Salmonella SAI", min_length=3)
    assert [(t.text, t.word, t.initial) for t in tokens] == [
        ("Sepsis", "sepsis", True),
        ("Salmonella", "salmonella", False),
        ("SAI", "sai", False),
    ]
    assert tokens[0].capitalized and tokens[1].capitalized


def test_build_keeps_capitals_written_away_from_the_first_word(tmp_path):
    cim10 = _normalized(
        tmp_path / "cim10.jsonl",
        [
            {"code": "A02.1", "source": "cim10", "label": "Sepsis à Salmonella", "synonyms": []},
            {"code": "A00.9", "source": "cim10", "label": "Choléra classique", "synonyms": []},
            {"code": "K50.9", "source": "cim10", "label": "Maladie de Crohn", "synonyms": []},
            {"code": "A01.0", "source": "cim10", "label": "Fièvre due au choléra", "synonyms": []},
        ],
    )
    output = tmp_path / "out.jsonl"

    build.build_words([("cim10", cim10)], output, RaritySettings(max_zipf=3.0))

    words = {json.loads(line)["word"] for line in output.read_text(encoding="utf-8").splitlines()}
    assert {"Salmonella", "Crohn", "choléra"} <= words
    assert "Choléra" not in words and "Sepsis" not in words
