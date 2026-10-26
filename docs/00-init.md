# 00 — Initialization and settings

## Install

Python 3.10–3.13 is required. Dependencies are declared in `pyproject.toml`. On Linux, `uv` obtains PyTorch from the ROCm index configured there; check that its ROCm version matches the cluster driver.

```bash
uv sync --extra train
uv run --extra train medstt --help
UV_PROJECT_ENVIRONMENT=.venv-tts uv sync --extra tts
```

If you use `pip`, install a PyTorch build suitable for the cluster before installing this project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm6.4
pip install -e '.[train]'
```

For TTS, create a separate `.venv-tts` and install `pip install -e '.[tts]'` there. On ROCm, install matching `torch` and `torchaudio` builds first. The two extras require different Transformers versions.

If compute nodes have no internet access, prepare the environment and download models on a connected machine before submitting jobs. See the [Slurm guide](slurm.md). Where home-directory quotas are small, put the `uv` and `pip` caches under `$SCRATCH` with `UV_CACHE_DIR=$SCRATCH/.cache/uv` and `PIP_CACHE_DIR=$SCRATCH/.cache/pip`.

## Create a workspace

```bash
export MEDSTT_WORKDIR=$SCRATCH/medstt-ws
medstt init "$MEDSTT_WORKDIR"
```

`init` creates the directories and `medstt.yaml` if they do not already exist. Other commands select the workspace in this order: `--workdir`, `MEDSTT_WORKDIR`, then the current directory. Relative input paths are searched first in the matching workspace subdirectory (for example, `words/cardio.txt`), then at the workspace root, then in the current directory.

```text
<workdir>/
  medstt.yaml       shared settings
  nomenclature/     source nomenclatures
  words/            term lists
  sentences/        sentence sets
  datasets/         audio datasets
  models/           local models
  runs/             training runs, metrics, final models
  evals/            future evaluation reports
  cache/hf/         Hugging Face cache
```

## `medstt.yaml`

`init` creates this file with the values below. Command-line options override the file, which overrides defaults in the code.

| Section | Key | Purpose / initial value |
| --- | --- | --- |
| `llm` | `base_url` | OpenAI-compatible endpoint, `http://localhost:8000/v1` |
| `llm` | `model` | LLM name, initially `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `llm` | `api_key_env`, `api_key` | Environment variable name (`MEDSTT_LLM_API_KEY`) and fallback key (`null`) |
| `llm` | `temperature`, `max_tokens` | Generation settings, `0.8` and `200` |
| `llm` | `concurrency`, `timeout` | Parallel requests (`16`) and request timeout in seconds (`120`) |
| `llm` | `extra_body` | Additional fields forwarded to the server, `{}` |
| `sentences` | `domain` | Prompt context (`pratique clinique hospitalière`); sentences are generated in French |
| `sentences` | `per_word` | Number of separate LLM requests and sentences for each term, `1` by default |
| `sentences` | `min_words`, `max_words` | Accepted sentence length, `8` to `20` words |
| `sentences` | `max_attempts` | Maximum attempts per sentence, `4` |
| `tts` | `voice_prompts.male`, `voice_prompts.female` | Two French descriptions for Qwen3-TTS VoiceDesign; one audio per description and sentence |
| `tts` | `hf_token` | Optional Hugging Face token (`null`); `HF_TOKEN` takes precedence |
| `tts` | `sampling_rate` | Output WAV sampling rate, `16000` Hz |
| `tts` | `min_duration`, `max_duration` | Accepted duration, `1.0` to `20.0` seconds |
| `train` | `run_name` | Name of the run directory under `runs/`, initially `cohere-medical` |
| `train` | `seed` | Random seed (`42`) for model training, dataset subsampling, and the evaluation split; any integer can be used |
| `train.data` | `train` | List of one or more training datasets; each `path` is a name under `datasets/` or an absolute path |
| `train.data` | `eval` | Optional list of separate evaluation datasets, initially `[]` |
| `train.data` | `eval_split` | Fraction of the combined training data held out for evaluation when `eval` is empty, initially `0.02` |
| `train.data` | `sampling_rate`, `max_label_tokens` | Audio rate (`16000` Hz) and maximum transcript length (`256` tokens) |
| `train.training` | `per_device_train_batch_size`, `per_device_eval_batch_size` | Examples processed per GPU at once, both `1` |
| `train.training` | `gradient_accumulation_steps` | Combine gradients from `8` batches before an optimizer update |
| `train.training` | `learning_rate`, `num_train_epochs`, `warmup_steps` | Learning rate (`0.00001`), passes over training data (`3`), and warmup steps (`200`) |
| `train.training` | `logging_steps`, `save_steps`, `eval_steps` | Log every `25` steps; save and evaluate every `500` steps |
| `train.training` | `save_total_limit`, `dataloader_num_workers` | Keep at most `3` checkpoints; use `4` data-loading workers |

To use several training datasets, add another `- path:` entry. They are combined before the optional evaluation split:

```yaml
train:
  run_name: cohere-medical
  seed: 42
  data:
    train:
      - path: synth-medical-words
      - path: synth-cardio
      - path: /absolute/path/to/another-dataset
    eval: []
    eval_split: 0.02
    sampling_rate: 16000
    max_label_tokens: 256
```

Each path must point to an [AudioFolder or saved Hugging Face dataset](datasets.md). The same `seed` gives repeatable random choices with the same data and setup; changing it can change the held-out samples and training order. See [Train](04-train.md) for the command and output files.

For the LLM endpoint, the API key comes from `--api-key`, then the variable named by `llm.api_key_env`, then `llm.api_key`. If none is set, the client sends `EMPTY`, which some local servers accept. Prefer an environment variable for a real key.

The [Sentences](02-sentences.md), [TTS](03-tts.md), and [Train](04-train.md) guides show how these settings are used. The trained model is saved under `runs/<run_name>/final/`.
