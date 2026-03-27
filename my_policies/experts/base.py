from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ..latents import RoutingLatents


class ActionHead(nn.Module, ABC):
    """Maps routed VLM-conditioned latents to action predictions."""

    @abstractmethod
    def forward(self, model, *, suffix_out: Tensor, routing_latents: RoutingLatents) -> Tensor:
        raise NotImplementedError


class RolloutPolicy(nn.Module, ABC):
    """Defines iterative action sampling behavior for an expert."""

    @abstractmethod
    def sample_actions(
        self,
        model,
        *,
        routing_latents: RoutingLatents,
        initial_noise: Tensor,
        action_head: ActionHead,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError


class TrainingObjective(nn.Module, ABC):
    """Computes expert-specific training loss from predictions and targets."""

    @abstractmethod
    def loss(self, prediction: Tensor, *, actions: Tensor, velocity_target: Tensor) -> Tensor:
        raise NotImplementedError


class ActionExpert(nn.Module):
    """Composable expert abstraction over head, rollout policy, and objective."""

    def __init__(
        self,
        *,
        expert_id: str,
        action_head: ActionHead,
        rollout_policy: RolloutPolicy,
        training_objective: TrainingObjective,
        supports_vendor_passthrough: bool = False,
    ) -> None:
        super().__init__()
        self.expert_id = expert_id
        self.action_head = action_head
        self.rollout_policy = rollout_policy
        self.training_objective = training_objective
        self.supports_vendor_passthrough = supports_vendor_passthrough

    def predict_from_suffix(self, model, *, suffix_out: Tensor, routing_latents: RoutingLatents) -> Tensor:
        return self.action_head(model, suffix_out=suffix_out, routing_latents=routing_latents)

    def training_loss(
        self,
        model,
        *,
        suffix_out: Tensor,
        routing_latents: RoutingLatents,
        actions: Tensor,
        velocity_target: Tensor,
    ) -> Tensor:
        prediction = self.predict_from_suffix(model, suffix_out=suffix_out, routing_latents=routing_latents)
        return self.training_objective.loss(prediction, actions=actions, velocity_target=velocity_target)

    def sample_actions(
        self,
        model,
        *,
        routing_latents: RoutingLatents,
        initial_noise: Tensor,
        **kwargs,
    ) -> Tensor:
        return self.rollout_policy.sample_actions(
            model,
            routing_latents=routing_latents,
            initial_noise=initial_noise,
            action_head=self.action_head,
            **kwargs,
        )

    def use_vendor_passthrough(self) -> bool:
        return self.supports_vendor_passthrough
