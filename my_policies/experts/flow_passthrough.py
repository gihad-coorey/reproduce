from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..ids import FLOW_EXPERT_ID
from ..latents import RoutingLatents
from .base import ActionExpert, ActionHead, RolloutPolicy, TrainingObjective


class FlowActionHead(ActionHead):
    """Thin action head matching vendor flow projection behavior."""

    def forward(self, model, *, suffix_out: Tensor, routing_latents: RoutingLatents) -> Tensor:
        del routing_latents
        return model.action_out_proj(suffix_out)


class EulerFlowRolloutPolicy(RolloutPolicy):
    """Vendor-aligned Euler rollout for flow velocity predictions."""

    def sample_actions(
        self,
        model,
        *,
        routing_latents: RoutingLatents,
        initial_noise: Tensor,
        action_head: ActionHead,
        **kwargs,
    ) -> Tensor:
        del action_head

        bsize = initial_noise.shape[0]
        device = initial_noise.device
        num_steps = model.config.num_steps
        dt = -1.0 / num_steps

        x_t = initial_noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return model.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=routing_latents.prefix_pad_masks,
                    past_key_values=routing_latents.past_key_values,
                    timestep=current_timestep,
                )

            if model._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = model.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if model.rtc_processor is not None and model.rtc_processor.is_debug_enabled():
                model.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t


class FlowVelocityMSEObjective(TrainingObjective):
    """Vendor-aligned flow objective over velocity targets."""

    def loss(self, prediction: Tensor, *, actions: Tensor, velocity_target: Tensor) -> Tensor:
        del actions
        return F.mse_loss(velocity_target, prediction, reduction="none")


class FlowPassthroughExpert(ActionExpert):
    """Thin wrapper expert for strict passthrough to vendor flow methods."""

    def __init__(self, strict_vendor_passthrough: bool = True) -> None:
        super().__init__(
            expert_id=FLOW_EXPERT_ID,
            action_head=FlowActionHead(),
            rollout_policy=EulerFlowRolloutPolicy(),
            training_objective=FlowVelocityMSEObjective(),
            supports_vendor_passthrough=strict_vendor_passthrough,
        )
