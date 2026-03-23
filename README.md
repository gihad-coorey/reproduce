# SmolVLA LIBERO Baseline

Reproducible baseline evaluation pipeline for SmolVLA on 5 LIBERO tasks.

## Eval Experiment Count Rundown

The eval script runs one rollout episode per task/seed and reports success rate per task.

Current defaults in `scripts/run_eval.py`:
- `EPISODES_PER_TASK = 4`
- `SMOKE_EPISODES = 1`
- `N_ENVS_PER_TASK = 4`
- Task list size from `configs/tasks.json` = 5 tasks

How to count total experiments:
- `total_experiments = number_of_tasks * episodes_per_task`

With current defaults:
- Normal mode (`python scripts/run_eval.py`): `5 * 4 = 20` total episodes/experiments
- Smoke mode (`python scripts/run_eval.py --smoke`): `1 * 1 = 1` total episode/experiment

Parallelization note:
- `N_ENVS_PER_TASK = 4` parallelizes rollout workers inside a task.
- It improves throughput, but does not change the total experiment count above.

## Headless Frame Dump (For Later Video Creation)

If your goal is smooth demo videos, headless frame dumping is usually faster and more stable than live GUI rendering.
The key reason is that no UI repaint loop is involved during rollout.

New options in `scripts/run_eval.py`:
- `--save-frames`: Save rendered PNG frames during eval every 5 steps (default when flag is present).
- `--save-frames <N>`: Override cadence and save one frame every `N` env steps.

Examples:
- Default cadence (every 5 steps):
	- `python scripts/run_eval.py --save-frames`
- Custom cadence (every 10 steps):
	- `python scripts/run_eval.py --save-frames 10`

Output layout:
- `results/headless_frame_dumps/<run_id>/<task_name>/episode_###/frame_#######.png`

Notes:
- Enabling `--save-frames` switches eval to a single environment per task (`n_envs=1`) for deterministic frame indexing.
- Writing PNG files adds disk I/O overhead. Increasing `N` in `--save-frames <N>` can reduce that cost.
