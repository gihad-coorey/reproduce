# Roadmap and Handoff Plan

Build a staged, low-confound implementation program that standardises expert interfaces first, adds FAST and VQ-VAE experts next, validates each expert through fixed routing ablations, then introduces learned routing only after independent expert competence is demonstrated.

## Steps

1. Phase 0: Baseline audit and contract freeze (blocks all later phases)
2. Phase 1: Define and stabilise a strict expert interface with shape/semantic invariants (depends on 1)
3. Phase 2: Implement FAST and VQ-VAE experts with independent adapters/tokenisation paths (depends on 2)
4. Phase 3: Add fixed-routing ablation policies for every expert and run inference-first integration validation (depends on 3)
5. Phase 4: Train each expert independently with shared data/backbone protocol and fail-fast gates (depends on 4)
6. Phase 5: Run subset-based non-learned multi-expert evaluation (depends on 5)
7. Phase 6: Implement and train learned MLP router with frozen experts, then compare to baselines (depends on 6)
8. Phase 7: Expand evaluation metrics (sample efficiency, generalisation, latency, completion efficiency) (parallel with late 6 once logging hooks exist)
9. Phase 8: Optional extensions only after stability criteria are met (depends on 7)

## Phase Details

### 1. Phase 0: Baseline audit and contract freeze

- Freeze the currently supported runtime policy names and behavior surface as baseline references.
- Snapshot current policy registry behavior and routing metadata emission points.
- Confirm default runtime invariants (seed, determinism, device policy) to avoid benchmark drift.
- Deliverable: short baseline checklist document under docs stating what is treated as fixed during the program.

### 2. Phase 1: Define and stabilise expert interface

- Introduce a strict, shared expert interface that all experts must satisfy.
- Required methods: `forward(latent)->action_chunk`, `tokenise(action_chunk)->tokens` (when applicable), `detokenise(tokens)->action_chunk`.
- Keep all experts externally shape-compatible: identical action chunk tensor shape and semantics expected by LIBERO action postprocessing.
- Add explicit shape contracts and validation helpers in one place; fail early with descriptive errors.
- Implement dummy/minimal mappings first (identity or simple linear) to validate tensor plumbing only.
- Deliverable: interface spec in docs plus unit tests for interface compliance and shape/semantic invariants.

### 3. Phase 2: Implement FAST and VQ-VAE experts

- Add FAST expert class with its own latent adapter, tokenisation, and detokenisation path.
- Add VQ-VAE expert class with codebook path (encoder-style tokeniser, decoder-style detokeniser) and same external interface.
- Do not force shared projection modules; allow per-expert adapter modules.
- Ensure both experts can execute end-to-end inference in the routed model wrapper.
- Deliverable: four runnable experts total: flow, binning, FAST, VQ-VAE.

### 4. Phase 3: Fixed routing ablations for pipeline validation

- Add fixed routers for AlwaysFAST and AlwaysVQVAE, mirroring existing AlwaysFlow and AlwaysBinning behavior.
- Register explicit runtime policies for each fixed router in policy registry.
- Validation sequence: inference-only smoke tests first, then minimal gradient-flow checks using tiny training loops.
- Confirm no hidden incompatibilities: scaling, normalization, temporal alignment, and output shape semantics.
- Deliverable: stable fixed-routing policy matrix that runs through CLI and GUI without interface conditionals.

### 5. Phase 4: Independent expert training

- Train each expert separately using identical dataset splits and frozen backbone constraints.
- Optimise reconstruction objective (initially MSE against ground-truth actions), with expert-specific auxiliary losses only when required.
- Record learning curves and final evaluation metrics in comparable format.
- Define fail criterion: experts that do not learn meaningful reconstruction/success are excluded from router training.
- Deliverable: per-expert training report and pass/fail inclusion decision.

### 6. Phase 5: Non-learned multi-expert evaluation

- Implement non-learned routing strategies: random, round-robin, plus simple heuristic if available.
- Evaluate selective subset matrix rather than full powerset: singles, selected pairs, full set.
- Prioritise pairs centered on flow plus one alternative (Flow+Binning, Flow+FAST, Flow+VQ-VAE).
- Deliverable: complementarity analysis showing whether representation diversity helps before learned routing.

### 7. Phase 6: Learned MLP router

- Implement MLP router taking routing latent as input and outputting expert logits/probabilities.
- Freeze experts initially; train router only.
- Support two supervision modes:
- mode A: indirect supervision via downstream trajectory error objective.
- mode B: offline best-expert labels per sample when feasible.
- Compare against fixed-routing and single-expert baselines using task success as primary metric.
- Deliverable: learned router benchmark report with ablations over supervision mode.

### 8. Phase 7: Extended evaluation metrics

- Add tracked metrics beyond task success: sample efficiency, held-out task generalisation, per-step latency, steps-to-completion.
- Ensure metric collection is consistent across single-expert, fixed-routing, and learned-routing runs.
- Deliverable: unified evaluation table and plots suitable for research reporting.

### 9. Phase 8: Optional extensions (time permitting)

- Joint fine-tuning of router and experts after router-only convergence baseline is established.
- Reinforcement learning-based router experiments.
- Scale-up runs to larger LIBERO coverage and episode budgets only after metric stability is demonstrated.
- Cross-architecture comparisons are optional and lower priority due to comparability concerns.

## Implementation Map (existing files to modify/reuse)

- `my_policies/experts/base.py`: add/standardise required tokenise and detokenise interface methods and shared invariants.
- `my_policies/experts/flow_passthrough.py`: adapt to the strict interface while preserving vendor passthrough path.
- `my_policies/experts/binning.py`: align to explicit tokenise/detokenise contract (currently soft decode path exists).
- `my_policies/routers/base.py`: preserve common router contract and metadata format for new fixed/learned routers.
- `my_policies/routers/fixed.py`: add AlwaysFAST and AlwaysVQVAE fixed routers.
- `my_policies/modeling.py`: keep routed orchestration point; ensure generic expert contract is used uniformly for training and sampling.
- `my_policies/__init__.py`: export new expert/router builders.
- `scripts/common.py`: extend policy registry with fixed-routing options and optionally learned-router variants.
- `scripts/run_eval.py`: ensure CLI choices and help text remain aligned with registry additions.
- `scripts/run_gui.py`: ensure GUI policy dropdown and metadata handling support additional policies.
- `docs/policies.md`: update policy semantics and interface guarantees.
- `docs/roadmap.md`: keep roadmap synchronized with implemented milestones.

## Expected New Modules (to add)

- New expert implementations under my_policies/experts for FAST and VQ-VAE.
- Optional learned router module under my_policies/routers for MLP routing logic.
- Training/evaluation support scripts for expert-only and router-only experiments (location to be chosen consistently with existing scripts layout).
- Test modules under project test area for expert interface compliance, fixed-routing integration, and minimal gradient-flow checks.

## Verification

1. Interface compliance tests: every expert passes forward/tokenise/detokenise contract checks and shape semantics checks.
2. Integration smoke tests: each fixed policy completes at least one task/episode inference pass in CLI mode.
3. Gradient sanity tests: tiny training step confirms nonzero grads for target trainable modules in each phase.
4. Baseline comparability checks: deterministic settings, seed policy, and task files are unchanged across ablations.
5. Training gate review: each expert must pass defined minimum learning criteria before inclusion in multi-expert routing phases.
6. Router evaluation checks: learned router must outperform or match selected non-learned routing baselines on primary metric with confidence intervals where possible.
7. Metrics consistency checks: latency and steps-to-completion are measured with identical protocol across policies.

## Decisions Recorded from User Direction

- Use phased progression with one new variable introduced at a time.
- Prioritise system correctness and comparability over early performance claims.
- Keep expert internals independent (no forced shared adapter) unless explicitly tested as an ablation.
- Do not expand optional extensions until all mandatory phases are stable.

## Scope Boundaries

- Included: interface standardisation, expert additions, fixed routing, independent training, learned routing, and expanded evaluation.
- Excluded until late: joint router+expert fine-tuning, RL router, large-scale benchmark expansion, cross-architecture comparisons.

## Execution Ordering and Parallelism Guidance

1. Sequential: Phases 0 to 4 should be strictly sequential.
2. Limited parallelism: in Phase 2, FAST and VQ-VAE implementation can proceed in parallel once interface is frozen.
3. Sequential gate: Phase 5 requires Phase 4 pass/fail decisions.
4. Parallel after instrumentation: Phase 7 metric tooling can begin in parallel with late Phase 6 evaluation once event/log schema is stable.
