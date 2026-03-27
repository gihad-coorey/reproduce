# SmolVLA LIBERO Baseline

Reproducible baseline evaluation pipeline for SmolVLA, from local smoke runs to full LIBERO-90 cluster evaluation.

## Eval Experiment Count Rundown

The eval script runs one rollout episode per task/seed and reports success rate per task.

Current defaults in `scripts/run_eval.py`:
- `EPISODES_PER_TASK = 10`
- `SMOKE_EPISODES = 1`
- `N_ENVS_PER_TASK = 4`
- Task list size from `configs/tasks.json` = 5 tasks

How to count total experiments:
- `total_experiments = number_of_tasks * episodes_per_task`

With current defaults:
- Normal mode (`python scripts/run_eval.py`): `5 * 10 = 50` total episodes/experiments
- Smoke mode (`python scripts/run_eval.py --smoke`): `1 * 1 = 1` total episode/experiment

Cluster scaling examples:
- Full task file with 10 episodes/task:
	- `python scripts/run_eval.py --policy "HF" --task-file configs/tasks_libero_90.json --episodes-per-task 10`
- Seeded run with explicit output artifact:
	- `python scripts/run_eval.py --policy "HF" --results-file results/official_eval_custom.json`

Full benchmark task configs:
- `configs/tasks.json`: small local subset.
- `configs/tasks_libero_90.json`: full LIBERO-90 task list (90 tasks).

Parallelization note:
- `N_ENVS_PER_TASK = 4` parallelizes rollout workers inside a task.
- It improves throughput, but does not change the total experiment count above.

## Runtime Policy Selection

The eval script supports runtime policy selection from a shared hardcoded registry in
`scripts/common.py` (`POLICY_REGISTRY`).

Current policy options:
- `HF`
- `FlowOnly`
- `BinOnly`

How to use it:
- Interactive prompt (default when no `--policy` is provided):
	- `python scripts/run_eval.py`
- Launch GUI from a standalone entry script:
	- `python scripts/run_gui.py`
- List configured options:
	- `python scripts/run_eval.py --list-policies`
- Select explicitly by name:
	- `python scripts/run_eval.py --policy FlowOnly`

Checkpoint behavior:
- The checkpoint is fixed to `HuggingFaceVLA/smolvla_libero` for all registered policies.
- The policy choice changes which class is instantiated, not which checkpoint path is used.

Non-interactive behavior:
- In non-interactive environments (for example, SLURM jobs), `--policy` is required.
- This prevents ambiguous default policy selection in batch runs.

## Fixed Runtime Contract

`scripts/run_eval.py` and `scripts/run_gui.py` now use an opinionated runtime policy.

Hardcoded behavior:
- Device selection is always `cuda -> mps`.
- CPU fallback requires an interactive confirmation prompt.
- If no CUDA/MPS is available in a non-interactive run (for example SLURM), evaluation fails fast.
- Seed is fixed to `0` in shared runtime code.
- Deterministic execution is always enabled.
- Runtime logging is always enabled (plain-text key=value format).

Remaining CLI controls:
- `--episodes-per-task N`: normal-mode episodes per task.
- `--n-envs N`: vectorized env count (non-frame mode).
- `--task-file PATH`: task-list JSON path.
- `--results-file PATH`: output results JSON path (optional override).
- `--frames-root PATH`: frame dump root (when `--save-frames` is enabled).

Default result grouping:
- If `--results-file` is omitted, outputs are written to:
	- `results/<policy>/<task_file_stem>/official_eval_<job_id>_<array_id>.json`

Environment variable equivalents:
- `LIBERO_EPISODES_PER_TASK`, `LIBERO_N_ENVS`
- `LIBERO_TASK_FILE`, `LIBERO_RESULTS_PATH`, `LIBERO_FRAMES_ROOT`
- `PYTORCH_ENABLE_MPS_FALLBACK`

## SLURM GPU Usage

Use the thin wrapper script and keep evaluation logic in `scripts/run_eval.py`.

Submission:
- `sbatch scripts/submit_eval_slurm.sh`

Recommended full benchmark seed-array submission:
- `sbatch scripts/submit_eval_slurm_array.sh`

Common overrides:
- `sbatch --export=CONDA_ENV_NAME=geng551x-gpu,EPISODES_PER_TASK=10,N_ENVS=4 scripts/submit_eval_slurm.sh`
- `sbatch --array=0-9 --export=CONDA_ENV_NAME=geng551x-gpu scripts/submit_eval_slurm_array.sh`

Array wrapper defaults:
- Uses `configs/tasks_libero_90.json`
- Uses `EPISODES_PER_TASK=10`
- Writes one result artifact per array index

Wrapper behavior:
- Activates conda env.
- Forces explicit policy/task/episodes/env count.
- Uses default grouped result outputs from `scripts/run_eval.py`:
	- `results/<policy>/<task_file_stem>/official_eval_<job_id>_<array_id>.json`

Why there are two SLURM scripts:
- `scripts/submit_eval_slurm.sh`: one standalone job (recommended for one task file per job).
- `scripts/submit_eval_slurm_array.sh`: scheduler fan-out for repeated runs across many array indices.

Runtime logging:
- Eval lifecycle events are emitted as canonical JSON Lines (one JSON object per event line) for robust post-run parsing.
- Human-readable key=value logs are still emitted for operator readability.
- Canonical events currently include `run_start`, `task_start`, `task_end`, and `run_end`.

Post-run aggregation from logs:
- Aggregate one or more SLURM logs:
	- `python scripts/aggregate_eval_logs.py --inputs "logs/**/*.out"`
- Write machine-readable summary JSON:
	- `python scripts/aggregate_eval_logs.py --inputs "logs/**/*.out" --output-json results/aggregates/latest.json --format json`
- Strict parsing mode (fail on malformed event JSON lines):
	- `python scripts/aggregate_eval_logs.py --inputs "logs/**/*.out" --strict`

To add more policies later, add a new entry to `POLICY_REGISTRY` with:
- `build_model`: a callable that loads and returns a policy instance

## Unified Eval Pipeline

The project now uses a 3-file eval layout for both CLI and GUI runs:

- Shared core/helpers/registry: `scripts/common.py`
- CLI entrypoint: `scripts/run_eval.py`
- GUI entrypoint: `scripts/run_gui.py`

Entrypoint policy:

- `scripts/run_eval.py` is CLI-only.
- Launch GUI via: `python scripts/run_gui.py`

This keeps policy loading, preprocessors, task execution, and event/schema behavior aligned across both modes while leaving GUI-specific rendering/UI logic in the GUI file.

## MyPolicy Architecture (Initial)

`my_policies` introduces a local policy stack without modifying vendored `external/lerobot` code.

Current architecture goals:
- Reuse one shared VLM prefix context path.
- Route from a transformer-processed prefix context latent.
- Dispatch to pluggable action experts.

Implemented components:
- `MyPolicy`: subclass of `SmolVLAPolicy` that defaults to `HuggingFaceVLA/smolvla_libero`.
- `RoutedVLAFlowMatching`: local model wrapper that extracts pooled prefix context latent and routes experts.
- `FlowOnlyRouter`: fixed route to native flow-matching expert.
- `BinningOnlyRouter`: fixed route to binning expert.

Notes:
- `MyPolicy FlowOnly` is intended as the behavior-preserving baseline path for parity checks.
- `MyPolicy BinningOnly` is an initial per-dim binning expert path that outputs continuous trajectories from soft bins.
- Router metadata (`router_impl`, `available_experts`) is emitted at run start for MyPolicy variants.

## Fine-Tuning and Experiment Plan

The project should be developed in stages so conclusions stay valid and regressions are easy to localize.

### Phase 1: Baseline and Parity Gates

1. Validate parity between `HF` and `FlowOnly`.
2. Run both smoke and full evaluation with fixed seeds and deterministic settings.
3. Track not only success rate but also trajectory-level diagnostics (for example per-episode action mean/std and per-dimension error summaries).
4. Treat parity failure as a blocker before enabling more complex routers or heads.

### Phase 2: Train Action Heads in Isolation

Train each action head independently first, using shared VLM context latents.

- Flow matching head:
	- Use as calibration baseline and sanity check.
	- Ensure no regression from pretrained behavior.

- Per-dim binning head:
	- Predict per-dimension bin logits and decode to continuous actions (expected bin centers).
	- Start with CE loss on bins plus an auxiliary continuous regression loss.
	- Begin with fixed bin ranges, then evaluate percentile-derived bin edges.

- VQ-VAE head:
	- Pretrain codebook on action chunks.
	- Train context-to-code predictor.
	- Decode to continuous trajectories and fine-tune.

- FAST head:
	- Train tokenized trajectory generation and decode back to continuous actions.
	- Monitor latency closely, not only quality.

### Phase 3: Router Training

Do not train the router without supervision.

1. Build oracle labels by evaluating all trained heads per sample and selecting the lowest-loss head.
2. Train router on pooled transformer-processed prefix context latent with cross-entropy objective.
3. Keep experts frozen during initial router training.
4. Optionally run a low-LR joint fine-tune stage after router convergence.

### Phase 4: Evaluation Matrix and Ablations

Evaluate every experiment under a fixed matrix:

1. `HF`
2. `FlowOnly`
3. `BinOnly`
4. Future multi-head router policies

Track:

- Online metrics: task success rate, mean/variance across seeds.
- Offline metrics: action MAE/MSE, smoothness/jerk statistics.
- Systems metrics: inference latency, throughput, memory footprint.

Run ablations for:

- Frozen vs partially unfrozen shared encoder.
- Different context pooling methods.
- Head-only vs router-only vs joint training.
- Per-chunk vs per-timestep routing (start with per-chunk).

### Phase 5: Experiment Hygiene

1. Keep `scripts/run_eval.py` and `scripts/run_gui.py` policy options and emitted router metadata in sync.
2. Log config/version identifiers in every result artifact.
3. Keep this README updated as each new head/router is implemented.
4. Update `/memories/repo/multi-head-router-roadmap.md` in the same change set as implementation updates.

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
