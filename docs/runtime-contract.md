# Runtime Contract

Shared runtime behavior lives in `scripts/common.py` and is used by both CLI and GUI.

## Fixed Behavior

- Model checkpoint is fixed to `HuggingFaceVLA/smolvla_libero`.
- Device selection order: CUDA, then MPS.
- CPU fallback requires interactive confirmation.
- In non-interactive mode without CUDA/MPS, runs fail fast.
- Seed is fixed to `0` in shared runtime usage.
- Deterministic mode is enabled by default.
- `PYTORCH_ENABLE_MPS_FALLBACK` is set to `1`.

## Policy Registry Contract

Current `POLICY_REGISTRY` entries:

- `HF`: upstream `SmolVLAPolicy` loaded from pretrained checkpoint
- `FlowOnly`: local `MyPolicy` with fixed flow-passthrough router
- `BinOnly`: local `MyPolicy` with fixed binning router

Adding new policies requires extending the registry in `scripts/common.py` with a `build_model` callable.

## Execution Modes

- Vectorized fast evaluation: `run_task_vectorized(...)`
- Stepwise evaluation (GUI and frame dump): `run_task_stepwise(...)`
