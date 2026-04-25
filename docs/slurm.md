# SLURM Usage

This repository currently provides one SLURM wrapper:

- `scripts/eval_slurm.sh`

## Submit

```bash
sbatch scripts/eval_slurm.sh
```

## Override Runtime Variables

```bash
sbatch --export=ALL,POLICY_NAME=HF,TASK_FILE=configs/tasks_libero_90.json,EPISODES_PER_TASK=10,N_ENVS=4 scripts/eval_slurm.sh
```

## Wrapper Behavior

- Loads Anaconda module
- Activates conda env `geng`
- Runs `python scripts/run_eval.py` with explicit `--policy`, `--task-file`, `--episodes-per-task`, `--n-envs`
- Writes SLURM logs to `logs/%j.out`
- Writes grouped eval outputs under `results/<sanitized_policy>/<sanitized_task_file_stem>/official_eval.json`

Current wrapper defaults (if not overridden in submit/export env vars):

- `POLICY_NAME=HF`
- `TASK_FILE=configs/tasks.json`
- `EPISODES_PER_TASK=8`
- `N_ENVS=4`

## Important Notes

- There is no array wrapper script in this repository at present.
- There is no built-in post-run log aggregation script at present.
