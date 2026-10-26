"""Failed generators must signal failure to dependent Slurm jobs."""

from importlib import import_module
from types import SimpleNamespace

from typer.testing import CliRunner

from medstt.cli import app
from medstt.workspace import Workspace


def test_sentence_failure_exits_nonzero(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "words" / "terms.txt").write_text("cardiologie\n", encoding="utf-8")
    monkeypatch.setattr("medstt.cli.generate_sentences", lambda *_args: (0, 1))

    result = CliRunner().invoke(
        app,
        ["sentences", "generate", "terms.txt", "-o", "cardio", "--workdir", str(ws.root)],
    )

    assert result.exit_code == 1
    assert "1 failed" in result.output


def test_tts_failure_exits_nonzero(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "sentences" / "cardio.txt").write_text(
        "Une phrase clinique en français.\n", encoding="utf-8"
    )

    def load_module(name):
        if name == "medstt.tts.backends":
            return SimpleNamespace(
                QwenVoiceDesign=lambda **_kwargs: object(),
                DEFAULT_VOICE_PROMPTS={"male": "m", "female": "f"},
            )
        return import_module(name)

    monkeypatch.setattr("medstt.cli.import_module", load_module)
    monkeypatch.setattr("medstt.tts.synthesize.synthesize_dataset", lambda *_args: (0, 0, 1))

    result = CliRunner().invoke(
        app,
        ["tts", "synthesize", "cardio.txt", "-o", "audio", "--workdir", str(ws.root)],
    )

    assert result.exit_code == 1
    assert "1 failed" in result.output
