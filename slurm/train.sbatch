#!/bin/bash
#SBATCH --job-name=medstt-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#
# Phase 3: fine-tune a speech-to-text model. One torchrun launcher per node
# starts one process per GPU (DDP over RCCL); scale with --nodes and --gpus-per-node.
#
# Usage (from the repository root):
#   sbatch -A <account> -p <partition> slurm/train.sbatch [--resume]
#
# Note: on MI250X nodes each GCD is exposed as a separate GPU (8 per node);
# on MI300A nodes each APU is one GPU (4 per node).

source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "[medstt] nodes=${SLURM_NNODES} gpus/node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

srun --kill-on-bad-exit=1 torchrun \
  --nnodes "${SLURM_NNODES}" \
  --nproc-per-node "${GPUS_PER_NODE}" \
  --rdzv-id "${SLURM_JOB_ID}" \
  --rdzv-backend c10d \
  --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  -m medstt train "$@"
