#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

mkdir -p logs results

# Runtime configuration knobs.
POLICY_NAME="${POLICY_NAME:-HF}"
TASK_FILE="${TASK_FILE:-configs/tasks.json}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-8}"
N_ENVS="${N_ENVS:-4}"
CONDA_ENV_NAME="geng"

module load Anaconda3
source /uwahpc/rocky9/python/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

echo "Running eval with policy=$POLICY_NAME episodes=$EPISODES_PER_TASK n_envs=$N_ENVS"
python scripts/run_eval.py \
  --policy "$POLICY_NAME" \
  --task-file "$TASK_FILE" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --n-envs "$N_ENVS"

echo "Run complete. Results should be in results/$POLICY_NAME"
