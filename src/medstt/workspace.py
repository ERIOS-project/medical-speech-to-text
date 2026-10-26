"""Create workspaces and resolve their settings and file paths.

A workspace holds ``medstt.yaml``, pipeline inputs and outputs, and a local
Hugging Face cache. Commands select it with ``--workdir``, then
``MEDSTT_WORKDIR``, then the current directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

WORKDIR_ENV = "MEDSTT_WORKDIR"
CONFIG_FILENAME = "medstt.yaml"
SUBDIRS = (
    "nomenclature",
    "words",
    "sentences",
    "datasets",
    "models",
    "runs",
    "evals",
    "cache/hf",
)

CONFIG_TEMPLATE = """\
# Workspace settings. Supported command-line options override matching values.

llm:
  # Any OpenAI-compatible endpoint (vLLM, TGI, LiteLLM, ...).
  base_url: http://localhost:8000/v1
  model: mistralai/Mistral-Small-3.2-24B-Instruct-2506
  # API key order: --api-key, the named environment variable, then api_key.
  # Prefer the environment variable for a real secret.
  api_key_env: MEDSTT_LLM_API_KEY
  api_key: null
  temperature: 0.8
  max_tokens: 200
  concurrency: 16
  timeout: 120
  # Extra request fields, for example to disable reasoning:
  # extra_body: {chat_template_kwargs: {enable_thinking: false}}
  extra_body: {}

sentences:
  domain: pratique clinique hospitalière
  # Number of different sentences requested for each term.
  per_word: 1
  min_words: 8
  max_words: 20
  max_attempts: 4

tts:
  # Fixed model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign.
  # Generate one recording per sentence with each of these two voice prompts.
  voice_prompts:
    male: >-
      Voix d'homme adulte francophone, naturelle, claire et posée,
      avec une diction médicale précise.
    female: >-
      Voix de femme adulte francophone, naturelle, claire et posée,
      avec une diction médicale précise.
  # Optional Hugging Face token. HF_TOKEN takes precedence when set.
  hf_token: null
  sampling_rate: 16000
  min_duration: 1.0
  max_duration: 20.0

train:
  # Train all weights of CohereLabs/cohere-transcribe-03-2026.
  # The trained model and processor are saved to runs/<run_name>/final/.
  run_name: cohere-medical
  seed: 42
  data:
    # Add one or more dataset names under datasets/, or absolute paths.
    train:
      - path: synth-medical-words
    # Separate evaluation datasets, or a fraction held out from train.
    eval: []
    eval_split: 0.02
    sampling_rate: 16000
    max_label_tokens: 256
  # Forwarded to transformers.Seq2SeqTrainingArguments.
  training:
    per_device_train_batch_size: 1
    per_device_eval_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 0.00001
    num_train_epochs: 3
    warmup_steps: 200
    logging_steps: 25
    save_steps: 500
    eval_steps: 500
    save_total_limit: 3
    dataloader_num_workers: 4
"""


class WorkspaceError(RuntimeError):
    """Report a missing workspace or invalid settings file."""


@dataclass(frozen=True)
class Workspace:
    """Workspace root and settings loaded from its ``medstt.yaml``."""

    root: Path
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | Path | None = None) -> Workspace:
        """Open an existing workspace and parse its settings."""
        root = _select_root(path)
        config_path = root / CONFIG_FILENAME
        if not config_path.is_file():
            raise WorkspaceError(
                f"No {CONFIG_FILENAME} in {root}. Run `medstt init {root}` first, "
                f"or point --workdir / ${WORKDIR_ENV} to an existing workspace."
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise WorkspaceError(f"{config_path} must contain a YAML mapping.")
        return cls(root=root, config=config)

    @classmethod
    def init(cls, path: str | Path | None = None) -> Workspace:
        """Create workspace folders and add default settings if absent."""
        root = _select_root(path)
        for name in SUBDIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
        config_path = root / CONFIG_FILENAME
        if not config_path.exists():
            config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        return cls.open(root)

    def section(self, name: str) -> dict[str, Any]:
        """Return a settings section as a mapping, or an empty mapping if absent."""
        value = self.config.get(name) or {}
        if not isinstance(value, dict):
            raise WorkspaceError(f"Section `{name}` of {CONFIG_FILENAME} must be a mapping.")
        return value

    def subdir(self, name: str) -> Path:
        """Return a workspace subdirectory, creating it if needed."""
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve(self, path: str | Path, base: str | None = None) -> Path:
        """Find an input under ``base``, the workspace root, or the current directory.

        Absolute paths are used as given. A path that does not exist is placed
        under the workspace's ``base`` directory when one is provided.
        """
        path = Path(path).expanduser()
        if path.is_absolute():
            return path
        candidates = []
        if base:
            candidates.append(self.root / base / path)
        candidates += [self.root / path, Path.cwd() / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def activate(self) -> None:
        """Select the workspace's HF cache unless ``HF_HOME`` is already set.

        Call this before loading libraries that read Hugging Face cache paths.
        """
        os.environ.setdefault("HF_HOME", str(self.root / "cache" / "hf"))


def _select_root(path: str | Path | None) -> Path:
    """Choose and normalize the workspace directory."""
    raw = path or os.environ.get(WORKDIR_ENV) or Path.cwd()
    return Path(raw).expanduser().resolve()


def pick(cli_value: Any, section: dict[str, Any], key: str, default: Any = None) -> Any:
    """Choose a CLI value, then a non-null setting, then the default."""
    if cli_value is not None:
        return cli_value
    value = section.get(key)
    return default if value is None else value
