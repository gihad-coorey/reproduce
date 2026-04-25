# SmolVLA LIBERO Baseline

Reproducible SmolVLA evaluation pipeline for LIBERO, with both CLI and GUI entrypoints and a local routed-policy extension stack for research experiments.

## Documentation Index

- [Getting Started](docs/getting-started.md)
- [CLI Evaluation](docs/evaluation-cli.md)
- [GUI Evaluation](docs/evaluation-gui.md)
- [Runtime Contract](docs/runtime-contract.md)
- [Policies](docs/policies.md)
- [SLURM Usage](docs/slurm.md)
- [Roadmap and Handoff Plan](docs/roadmap.md)

## Quick Start

1. Create environment:

```bash
conda env create -n geng -f conda.yml
conda activate geng
```

2. Download LIBERO dataset:

```bash
python scripts/download_libero.py
```

3. Run smoke evaluation:

```bash
python scripts/run_eval.py --policy HF --smoke
```

4. Optional GUI:

```bash
python scripts/run_gui.py
```

## Repository Intent

- Baseline-first evaluation with reproducible runtime settings
- Side-by-side policy comparisons (`HF`, `FlowOnly`, `BinOnly`)
- Safe local experimentation in `my_policies` without editing vendored `external/lerobot`
