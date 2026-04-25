# CLI Evaluation

The CLI entrypoint is `scripts/run_eval.py`.

## Policy Selection

The runtime registry currently exposes:

- `HF`
- `FlowOnly`
- `BinOnly`

Useful commands:

```bash
python scripts/run_eval.py --list-policies
python scripts/run_eval.py --policy HF --task-file configs/tasks.json
python scripts/run_eval.py --policy FlowOnly --smoke
python scripts/run_eval.py --policy BinOnly --episodes-per-task 10 --n-envs 4
```

## Core CLI Arguments

- `--policy`: policy name from registry
- `--smoke`: first task only, one episode
- `--episodes-per-task N`: episodes per task in full mode
- `--n-envs N`: vectorized env count for non-frame runs
- `--task-file PATH`: task list JSON path
- `--save-frames` or `--save-frames N`: save headless PNG frames every N steps

## Current Runtime Defaults

From shared constants and argument wiring:

- `episodes_per_task = 10`
- `smoke_episodes = 1`
- `n_envs = 1` (unless overridden by `--n-envs` or `N_ENVS`)
- default task file: `configs/tasks.json`

## Experiment Counting

The evaluator runs one rollout episode per task/seed and reports per-task success rate.

Formula:

- `total_episodes = number_of_tasks * episodes_per_task`

With defaults (`configs/tasks.json`, 5 tasks, 10 episodes/task):

- Full mode: `5 * 10 = 50` episodes
- Smoke mode: `1 * 1 = 1` episode

`--n-envs` changes throughput, not total episode count.

## Outputs

Default output grouping:

- `results/<sanitized_policy>/<sanitized_task_file_stem>/official_eval.json`
- Event log JSONL alongside results:
  - `results/<sanitized_policy>/<sanitized_task_file_stem>/official_eval.jsonl`

When `--save-frames` is enabled:

- PNG frames are written to:
  - `results/headless_frame_dumps/<run_id>/<task_name>/episode_###/frame_#######.png`
- Runtime forces `n_envs=1` for deterministic frame indexing.

## Runtime Events

The CLI emits structured lifecycle events (JSONL), including:

- `run_start`
- `task_end`
- `run_end`
- `step_update` (stepwise mode)
