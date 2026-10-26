#!/bin/bash
#SBATCH --job-name=medstt-tts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-logs/%x-%A_%a.out
#
# Phase 2: synthesize two audio files per sentence. Each array task handles one
# shard of the sentences; resubmitting skips what is already done.
#
# Usage (from the repository root):
#   jid=$(sbatch --parsable -A <account> -p <partition> --array=0-7 \
#           slurm/synthesize.sbatch <sentences-file> <dataset-name> [medstt options])
#   sbatch -A <account> -p <partition> --dependency=afterok:${jid} \
#           slurm/finalize_dataset.sbatch <dataset-name>

export MEDSTT_VENV="${MEDSTT_VENV:-${SLURM_SUBMIT_DIR}/.venv-tts}"
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

SENTENCES=${1:?usage: synthesize.sbatch <sentences-file> <dataset-name> [options]}
OUTPUT=${2:?usage: synthesize.sbatch <sentences-file> <dataset-name> [options]}
shift 2

medstt tts synthesize "${SENTENCES}" --output "${OUTPUT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
  "$@"
