# Common environment for medstt Slurm jobs, sourced by every slurm/*.sbatch script.
# Adapt the "module" lines to your cluster (`module avail rocm`).

set -euo pipefail

: "${MEDSTT_WORKDIR:?Export MEDSTT_WORKDIR (the medstt workspace) before calling sbatch.}"
REPO_DIR="${MEDSTT_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDSTT_VENV="${MEDSTT_VENV:-${REPO_DIR}/.venv}"

# module purge
# module load rocm/6.4

# shellcheck disable=SC1091
source "${MEDSTT_VENV}/bin/activate"

# Compute nodes have no internet access: everything comes from the workspace cache.
export HF_HOME="${HF_HOME:-${MEDSTT_WORKDIR}/cache/hf}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# MIOpen (ROCm) kernel databases must live on node-local storage: SQLite locking
# breaks on shared filesystems and concurrent jobs would corrupt them.
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/miopen-${USER}-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Uncomment on MI300A APUs to let the GPU page in host memory (unified memory).
# export HSA_XNACK=1

echo "[medstt] job=${SLURM_JOB_ID:-local} node=$(hostname) workdir=${MEDSTT_WORKDIR}"
echo "[medstt] python=$(command -v python) torch=$(python -c 'import torch; print(torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count())')"
