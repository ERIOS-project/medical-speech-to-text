# Run the steps on Slurm

Submit the scripts in [`slurm/`](../slurm) from the repository root. Set `MEDSTT_WORKDIR` and adapt the ROCm module setup in `slurm/env.sh` for your cluster. Jobs activate `.venv` by default; the synthesis script selects `.venv-tts`. The shared setup enables Hugging Face offline mode and places MIOpen caches on node-local storage. Supply your account and partition to `sbatch`; logs go to `slurm-logs/`.

```bash
export MEDSTT_WORKDIR=$SCRATCH/medstt-ws
```

Prepare [models and datasets](datasets.md) on a connected machine before submitting these jobs. Install the TTS extra in `.venv-tts` and the training extra in `.venv`; the synthesis script activates `.venv-tts` automatically.

## Sentences

With an LLM endpoint already reachable from the job:

```bash
sbatch -A <account> -p <partition> \
  slurm/generate_sentences.sbatch cardio.txt cardio
```

The script can also start vLLM in the job when `MEDSTT_VLLM_MODEL` is set. The model must already be available locally or in the cache. Set `MEDSTT_VLLM_BIN` if needed and request the required GPUs with `sbatch --gpus-per-node=<count>`.

## Audio

A job array splits sentence/voice pairs into shards. The dependent job merges their metadata after all shards succeed. A generation error now makes its shard fail, so Slurm will not finalize incomplete output; duration rejections are recorded separately.

```bash
jid=$(sbatch -A <account> -p <partition> --parsable --array=0-7 \
  slurm/synthesize.sbatch cardio.jsonl synth-cardio)
sbatch -A <account> -p <partition> --dependency=afterok:$jid \
  slurm/finalize_dataset.sbatch synth-cardio
```

## Training

The script uses `torchrun` with one process per GPU and supports multiple nodes:

```bash
sbatch -A <account> -p <partition> --nodes=1 --gpus-per-node=4 \
  slurm/train.sbatch
```

Training reads the `train` section of the workspace's `medstt.yaml`. To resume interrupted training, pass `--resume` after `slurm/train.sbatch`. Sentence generation and audio synthesis skip items already recorded.
