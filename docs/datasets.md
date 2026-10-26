# Prepare models and datasets

Run these commands on a machine with internet access before submitting jobs to offline compute nodes:

```bash
medstt model download CohereLabs/cohere-transcribe-03-2026
medstt model download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
medstt dataset import facebook/multilingual_librispeech \
  --config french --split train --text-column transcript \
  --max-samples 20000 -o mls-fr-train
```

`model download` saves both fixed model IDs in the workspace's Hugging Face cache. Keep the default cache location so training and TTS can load them by Hub ID. `dataset import` copies a Hugging Face dataset or a compatible local dataset into `datasets/<name>/` as an AudioFolder. Adapt the import with `--audio-column`, `--text-column`, `--config`, `--split`, `--max-samples`, and `--streaming`. Import writes the audio files and `metadata.csv` without decoding audio during the copy.

For manually prepared data, the loader accepts an AudioFolder directory with `metadata.csv` or `metadata.jsonl` containing `file_name` and `text`, or a directory created by `Dataset.save_to_disk`. In `medstt.yaml`, each `train.data.train` or `train.data.eval` item can be a dataset name or an object with `path`, `split`, `max_samples`, `repeat`, and column names. `medstt dataset info <name>` prints the sample count and a few transcripts.

If downloads were done on another machine, copy the workspace to the cluster. Compute nodes use its `cache/hf/` directory.
