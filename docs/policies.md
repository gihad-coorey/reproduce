# Policy Implementations

This project evaluates three runtime policy options from a shared registry.

## Summary

- `HF`: upstream SmolVLA policy class, no local routing layer.
- `FlowOnly`: local routed policy that selects a flow expert and then delegates to vendored flow behavior.
- `BinOnly`: local routed policy that selects a binning expert with a different action head/objective path.

## HF

`HF` builds `SmolVLAPolicy.from_pretrained(OFFICIAL_MODEL_ID)`.

Characteristics:

- Uses vendored SmolVLA implementation directly.
- No local router/expert abstraction.
- Intended baseline for comparison.

## FlowOnly

`FlowOnly` builds `MyPolicy.from_pretrained(..., router="flow_only")`.

Implementation path:

- Wraps model internals with `RoutedVLAFlowMatching`.
- Router is fixed (`FlowOnlyRouter`) and always chooses flow expert.
- Flow expert supports strict vendored passthrough:
  - Training delegates to vendored `VLAFlowMatching.forward(...)`.
  - Sampling delegates to vendored `VLAFlowMatching.sample_actions(...)`.

Implication:

- Architecture adds routing instrumentation and metadata, but action generation path is intentionally parity-oriented with upstream flow behavior.

## BinOnly

`BinOnly` builds `MyPolicy.from_pretrained(..., router="binning_only")`.

Implementation path:

- Uses same routed wrapper and shared prefix latent extraction.
- Router is fixed (`BinningOnlyRouter`) and always chooses binning expert.
- Binning expert uses a learned binning head:
  - Projects suffix hidden states to per-dimension logits over bins.
  - Decodes continuous action via expected bin center.
  - Uses direct action MSE objective for training path.

Sampling differences vs flow path:

- Uses `denoise_step_hidden(...)` to obtain hidden states each step.
- Recomputes actions from binning head rather than vendored flow velocity head.

## Router Metadata

For local routed policies, run metadata includes:

- `router_impl`
- `available_experts`
- latest `router_choice`, confidence, and route metadata at task level
