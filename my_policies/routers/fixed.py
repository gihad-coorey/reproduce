from __future__ import annotations

from ..experts import BinningActionExpert, FlowPassthroughExpert
from ..ids import BINNING_EXPERT_ID, FLOW_EXPERT_ID
from ..latents import RoutingLatents
from .base import ActionExpertRouter, RouterDecision


class _SingleExpertRouter(ActionExpertRouter):
    """Fixed router that always selects one registered expert."""

    def __init__(self, *, expert, mode: str):
        super().__init__({expert.expert_id: expert})
        self._expert_id = expert.expert_id
        self._mode = mode

    def route(self, routing_latents: RoutingLatents) -> RouterDecision:
        del routing_latents
        return RouterDecision(expert_id=self._expert_id, confidence=1.0, metadata={"mode": self._mode})


class FlowOnlyRouter(_SingleExpertRouter):
    """Router with a registry containing only the flow passthrough expert."""

    def __init__(self) -> None:
        super().__init__(expert=FlowPassthroughExpert(strict_vendor_passthrough=True), mode="fixed_flow")


class BinningOnlyRouter(_SingleExpertRouter):
    """Router with a registry containing only the binning expert."""

    def __init__(self, *, hidden_size: int, max_action_dim: int, n_bins: int = 64) -> None:
        super().__init__(
            expert=BinningActionExpert(hidden_size=hidden_size, max_action_dim=max_action_dim, n_bins=n_bins),
            mode="fixed_binning",
        )
