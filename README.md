# MedSTT — Medical Speech Recognition

This project builds a synthetic speech dataset from medical terms and fine-tunes [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) on that vocabulary. It is designed for Slurm clusters with AMD GPUs (ROCm), including compute nodes without internet access.

## Workflow

| Step | What happens | Output |
| --- | --- | --- |
| [00 — Initialize](docs/00-init.md) | Create the workspace and configure `medstt.yaml`. | Workspace directories and settings |
| [01 — Medical terms](docs/01-words.md) | Provide a term list. **TODO:** extract terms from a nomenclature. | `words/` |
| [02 — Sentences](docs/02-sentences.md) | Ask an LLM to write and validate a sentence for each term. | `sentences/*.jsonl` |
| [03 — Synthetic speech](docs/03-tts.md) | Generate one male and one female recording per sentence with Qwen3-TTS VoiceDesign. | `datasets/` (WAV files and metadata) |
| [04 — Training](docs/04-train.md) | Train all Cohere Transcribe weights, optionally mixing in existing French speech datasets. | `runs/<run_name>/final/` (model and processor) |
| [05 — Evaluation](docs/05-evaluate.md) | **TODO:** standalone evaluation. Metrics are currently computed during training. | Training evaluation metrics |

You can start with a manually prepared term list, sentence file, or audio dataset without running the preceding steps.

## Requirements

- Python 3.10–3.13.
- Linux x86_64 with a compatible AMD ROCm driver for GPU training, or macOS for local development. Check the ROCm index in `pyproject.toml` against your cluster's driver.
- Access to an OpenAI-compatible LLM endpoint for sentence generation. Audio synthesis uses the local Qwen3-TTS VoiceDesign model through its Python library.
- A Hugging Face account with access to the gated Cohere Transcribe model, plus a token for downloading it before offline training.

## Installation

With `uv` (recommended):

```bash
uv sync --extra train
uv run --extra train medstt --help
UV_PROJECT_ENVIRONMENT=.venv-tts uv sync --extra tts
```

With `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On a Linux AMD cluster, install the ROCm builds first (adjust the index to match your driver):

```bash
pip install torch pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm6.4
```

Then install the project with the training dependencies:

```bash
pip install -e '.[train]'
medstt --help
```

For TTS, use a second environment with `pip install -e '.[tts]'`. Qwen3-TTS currently requires a different Transformers version from Cohere Transcribe, so the two extras cannot be installed together. On a ROCm cluster, install the matching PyTorch and torchaudio builds in that environment first.

Initialize a workspace with `uv run --extra train medstt init /path/to/workspace` (or `medstt init /path/to/workspace` in the activated `pip` environment). See the [step-by-step documentation](docs/index.md) for inputs, commands, outputs, and settings.

## Development checks

### Lint and formatting

Ruff checks `src/` and `tests/` for Python errors, unused or unsorted imports, common bugs, and unnecessary complexity. It also enforces docstring style in `src/`. Its formatter checks the layout of both directories. Run the checks with:

```bash
make lint
```

To apply Ruff's safe fixes and formatting, run `make format`. The lint checks also run on pushes and pull requests through GitHub Actions.

### Tests

Run the training and pipeline tests in the training environment:

```bash
uv run --extra train pytest -q
```

Run the Qwen wrapper tests in the separate TTS environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-tts uv run --extra tts pytest -q tests/test_qwen_tts.py
```

The Qwen tests are skipped by the training-environment command and run by the second command.

These tests use synthetic audio, mocks, and a small Cohere model for one CPU training step. They need no GPU or downloaded model weights. The full model checkpoints, actual Qwen audio generation, and multi-GPU Slurm jobs must be checked on the target cluster.

## License and attribution

The project is developed by CHU de Montpellier and its source code is licensed under the [Apache License 2.0](LICENSE). Distributions of the project and derivative works must retain the applicable attribution in [NOTICE](NOTICE).

When citing the project in a publication or presentation, please use: **CHU de Montpellier. MedSTT — Medical Speech Recognition.** Include the [project URL](https://github.com/ERIOS-project/medical-speech-to-text) where possible.
