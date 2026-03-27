from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import nn

from ..experts import ActionExpert
from ..latents import RoutingLatents


@dataclass
class RouterDecision:
    expert_id: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionExpertRouter(nn.Module, ABC):
    """Router interface with an explicit local expert registry."""

    def __init__(self, experts: Mapping[str, ActionExpert]):
        super().__init__()
        if len(experts) == 0:
            raise ValueError("ActionExpertRouter requires at least one expert in its registry.")
        self.experts = nn.ModuleDict(dict(experts))

    @property
    def available_experts(self) -> list[str]:
        return list(self.experts.keys())

    def get_expert(self, expert_id: str) -> ActionExpert:
        if expert_id not in self.experts:
            available = ", ".join(self.available_experts)
            raise ValueError(f"Router selected unknown expert '{expert_id}'. Available: {available}")
        return self.experts[expert_id]

    @abstractmethod
    def route(self, routing_latents: RoutingLatents) -> RouterDecision:
        raise NotImplementedError
