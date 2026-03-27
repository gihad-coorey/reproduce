#!/bin/bash
#SBATCH --job-name=libero-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/libero_eval_%j.out

# Single-job launcher: submit one full evaluation run per task file.
# Use submit_eval_slurm_array.sh when you want scheduler-managed fan-out.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

mkdir -p logs results

# Update these defaults via environment variables when submitting with sbatch.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geng551x-gpu}"
POLICY_NAME="${POLICY_NAME:-HF}"
TASK_FILE="${TASK_FILE:-configs/tasks_libero_90.json}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
N_ENVS="${N_ENVS:-4}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Load your cluster conda module first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "Running eval with policy=$POLICY_NAME episodes=$EPISODES_PER_TASK n_envs=$N_ENVS"
python scripts/run_eval.py \
  --policy "$POLICY_NAME" \
  --task-file "$TASK_FILE" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --n-envs "$N_ENVS"

echo "Run complete. Results are grouped under results/<policy>/<task_file_stem>/."
