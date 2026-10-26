#!/bin/bash
#SBATCH --job-name=medstt-sentences
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#
# Phase 1: generate one sentence per word with an LLM.
#
# Usage (from the repository root):
#   sbatch -A <account> -p <partition> slurm/generate_sentences.sbatch <words-file> <output-name> [medstt options]
#
# The LLM endpoint comes from medstt.yaml (llm.base_url). Since compute nodes are
# offline, the job can instead start a local vLLM server on the allocated GPUs:
#   MEDSTT_VLLM_MODEL=<hub id or local path in the HF cache> \
#   MEDSTT_VLLM_BIN=/path/to/vllm-env/bin/vllm \
#   sbatch --gpus-per-node=4 ... slurm/generate_sentences.sbatch words.txt my-sentences
# Without MEDSTT_VLLM_MODEL, the job uses an external LLM and requests no GPU.

source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

WORDS=${1:?usage: generate_sentences.sbatch <words-file> <output-name> [options]}
OUTPUT=${2:?usage: generate_sentences.sbatch <words-file> <output-name> [options]}
shift 2

EXTRA=()
if [[ -n "${MEDSTT_VLLM_MODEL:-}" ]]; then
  GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-0}
  if (( GPUS_PER_NODE < 1 )); then
    echo "MEDSTT_VLLM_MODEL requires GPUs; submit with --gpus-per-node=<count>." >&2
    exit 1
  fi
  PORT=$((20000 + SLURM_JOB_ID % 10000))
  "${MEDSTT_VLLM_BIN:-vllm}" serve "${MEDSTT_VLLM_MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size "${GPUS_PER_NODE}" \
    ${MEDSTT_VLLM_ARGS:-} \
    > "slurm-logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT
  echo "[medstt] waiting for vLLM on port ${PORT}..."
  until curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; do
    kill -0 "${VLLM_PID}" 2>/dev/null || { echo "vLLM exited, see slurm-logs/vllm-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 10
  done
  EXTRA=(--base-url "http://127.0.0.1:${PORT}/v1" --model "${MEDSTT_VLLM_MODEL}")
fi

medstt sentences generate "${WORDS}" --output "${OUTPUT}" "${EXTRA[@]}" "$@"
