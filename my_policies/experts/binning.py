from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..ids import BINNING_EXPERT_ID
from ..latents import RoutingLatents
from .base import ActionExpert, ActionHead, RolloutPolicy, TrainingObjective


class BinningActionHead(ActionHead):
    """Project-specific binning head mapping suffix latents to continuous actions."""

    def __init__(self, hidden_size: int, max_action_dim: int, n_bins: int = 64) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.max_action_dim = max_action_dim
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        self.bin_head = nn.Linear(hidden_size, max_action_dim * n_bins)
        self.register_buffer("bin_centers", torch.linspace(-1.0, 1.0, n_bins), persistent=False)

    def forward(self, model, *, suffix_out: Tensor, routing_latents: RoutingLatents) -> Tensor:
        del model
        context_bias = self.context_proj(routing_latents.context_latent)[:, None, :]
        fused = suffix_out + context_bias
        logits = self.bin_head(fused)
        bsize, chunk_size, _ = logits.shape
        logits = logits.view(bsize, chunk_size, self.max_action_dim, self.n_bins)
        probs = torch.softmax(logits, dim=-1)
        centers = self.bin_centers.to(dtype=probs.dtype, device=probs.device)
        return torch.sum(probs * centers, dim=-1)


class DirectActionRolloutPolicy(RolloutPolicy):
    """Rollout that predicts direct actions from hidden suffix latents."""

    def sample_actions(
        self,
        model,
        *,
        routing_latents: RoutingLatents,
        initial_noise: Tensor,
        action_head: ActionHead,
        **kwargs,
    ) -> Tensor:
        del kwargs

        bsize = initial_noise.shape[0]
        device = initial_noise.device
        num_steps = model.config.num_steps
        dt = -1.0 / num_steps

        x_t = initial_noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            suffix_out = model.denoise_step_hidden(
                x_t=x_t,
                prefix_pad_masks=routing_latents.prefix_pad_masks,
                past_key_values=routing_latents.past_key_values,
                timestep=time_tensor,
            )
            x_t = action_head(model, suffix_out=suffix_out, routing_latents=routing_latents)
        return x_t


class ActionMSEObjective(TrainingObjective):
    """Direct action supervision objective for non-flow experts."""

    def loss(self, prediction: Tensor, *, actions: Tensor, velocity_target: Tensor) -> Tensor:
        del velocity_target
        return F.mse_loss(actions, prediction, reduction="none")


class BinningActionExpert(ActionExpert):
    """Binning expert conditioned on shared VLM routing latents."""

    def __init__(self, hidden_size: int, max_action_dim: int, n_bins: int = 64) -> None:
        super().__init__(
            expert_id=BINNING_EXPERT_ID,
            action_head=BinningActionHead(hidden_size=hidden_size, max_action_dim=max_action_dim, n_bins=n_bins),
            rollout_policy=DirectActionRolloutPolicy(),
            training_objective=ActionMSEObjective(),
            supports_vendor_passthrough=False,
        )
