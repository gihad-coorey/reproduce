from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RoutingLatents:
    """Shared routing payload derived from VLM prefix latents."""

    context_latent: Tensor
    prefix_pad_masks: Tensor
    past_key_values: object | None = None


@dataclass
class ExpertTrainingBatch:
    """Training payload consumed by action experts."""

    routing: RoutingLatents
    suffix_out: Tensor
    actions: Tensor
    velocity_target: Tensor
