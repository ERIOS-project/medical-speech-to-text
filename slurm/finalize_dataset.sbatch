#!/bin/bash
#SBATCH --job-name=medstt-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/%x-%j.out
#
# Merge the shards written by synthesize.sbatch into the dataset metadata.csv.
# Usage: sbatch -A <account> -p <partition> --dependency=afterok:<jid> slurm/finalize_dataset.sbatch <dataset-name>

source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

medstt dataset finalize "${1:?usage: finalize_dataset.sbatch <dataset-name>}"
