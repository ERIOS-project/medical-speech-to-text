# 03 — Generate synthetic speech

This step turns a [sentence file](02-sentences.md) into an [AudioFolder dataset](datasets.md). It uses the Python `qwen-tts` library directly with the fixed [Qwen3-TTS-12Hz-1.7B-VoiceDesign model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign). Each sentence is synthesized **once with a male voice and once with a female voice**. There is no audio repeat setting or TTS endpoint.

## Install the TTS environment

Qwen3-TTS and Cohere Transcribe require different Transformers versions. From the repository root, install the TTS extra in its own environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-tts uv sync --extra tts
```

With `pip`:

```bash
python3 -m venv .venv-tts
source .venv-tts/bin/activate
pip install -e '.[tts]'
```

On a ROCm cluster, install matching PyTorch and torchaudio builds before the last command; see [installation](../README.md#installation).

## Configure the two voices

New workspaces created by `medstt init` include this `tts` section in `<workdir>/medstt.yaml`. For an existing workspace, add or replace its `tts` section:

```yaml
tts:
  voice_prompts:
    male: >-
      Voix d'homme adulte francophone, naturelle, claire et posée,
      avec une diction médicale précise.
    female: >-
      Voix de femme adulte francophone, naturelle, claire et posée,
      avec une diction médicale précise.
  hf_token: null
  sampling_rate: 16000
  min_duration: 1.0
  max_duration: 20.0
```

The two prompts describe the desired voices; both keys are required. Qwen receives `language="French"` and the matching prompt for each recording. `hf_token` is optional, and the `HF_TOKEN` environment variable takes precedence. The model name, language, and number of recordings per sentence are fixed.

## Synthesize

Select the workspace with `MEDSTT_WORKDIR` or `--workdir`. For example, with `<workdir>/sentences/cardio.jsonl`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-tts uv run --extra tts medstt tts synthesize cardio.jsonl -o synth-cardio
```

In an activated `pip` TTS environment, run `medstt tts synthesize cardio.jsonl -o synth-cardio`. The model weights are downloaded on first use. For an offline compute node, [download the model into the workspace cache beforehand](datasets.md).

The command writes `<workdir>/datasets/synth-cardio/`:

```text
metadata.csv
audio/000000-0-male.wav
audio/000000-0-female.wav
_parts/part-00000.jsonl
_parts/rejected-00000.jsonl  # only if an audio file was rejected
```

For each successful WAV, `metadata.csv` has one row with its `file_name`, transcript (`text`), `speaker` (`male` or `female`), and original sentence ID (`source_id`). Audio is converted to mono, peak-normalized, and saved as 16 kHz PCM WAV by default. Files outside the configured duration range are rejected, so the final dataset can contain fewer than two rows for a sentence.

Rerunning the command skips completed sentence/voice pairs and retries failed generations without making another copy of a successful voice. With one shard, `metadata.csv` is finalized automatically. For Slurm arrays, each task uses `--shard-index` and `--num-shards`; run `medstt dataset finalize synth-cardio` when all tasks finish. See [Slurm usage](slurm.md#audio).
