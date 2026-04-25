# Getting Started

This repository provides a reproducible evaluation pipeline for SmolVLA on LIBERO tasks, with both CLI and GUI entrypoints.

## 1. Create/Update Environment

Use the conda environment file:

```bash
conda env create -n geng -f conda.yml
# or
conda env update -n geng -f conda.yml --prune
```

Then activate:

```bash
conda activate geng
```

Key dependencies are declared in `conda.yml`, including `lerobot`, `libero`, `imageio`, and Hugging Face Hub tooling.

## 2. Download LIBERO Dataset

Use the helper script:

```bash
python scripts/download_libero.py
```

This pulls the dataset into `data/libero`.

## 3. Verify Task Files

Current task config sizes:

- `configs/tasks.json`: 5 tasks (local subset)
- `configs/tasks_libero_10.json`: 10 tasks
- `configs/tasks_libero_90.json`: 90 tasks
- `configs/tasks_libero_goal.json`: 10 tasks
- `configs/tasks_libero_object.json`: 10 tasks
- `configs/tasks_libero_spatial.json`: 10 tasks

## 4. First Smoke Run

```bash
python scripts/run_eval.py --policy HF --smoke
```

If no `--policy` is provided in an interactive terminal, the script prompts for one. In non-interactive runs, `--policy` is required.

## 5. Optional GUI Run

```bash
python scripts/run_gui.py
```

Select tasks/policy in the UI, then run evaluation with live preview.
