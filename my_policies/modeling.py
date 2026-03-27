from __future__ import annotations

"""Routed SmolVLA model extensions with provenance annotations.

Vendor baseline:
- Core architecture, embeddings, attention mask construction, denoising math,
  and policy preprocessing flow come from
  `lerobot.policies.smolvla.modeling_smolvla`.

Our augmentations:
- Introduce a latent-first orchestration path:
  RoutedVLAFlowMatching -> VLM-derived latents -> Router -> ActionExpert.
- Keep strict passthrough support to vendored flow behavior through a thin
  flow expert wrapper.

Why:
- Preserve upstream compatibility while enabling multiple action experts that
  consume the same routed VLM latent contract.
"""

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks

from .latents import ExpertTrainingBatch, RoutingLatents
from .router import ActionExpertRouter, BinningOnlyRouter, FlowOnlyRouter

OFFICIAL_MODEL_ID = "HuggingFaceVLA/smolvla_libero"


class RoutedVLAFlowMatching(VLAFlowMatching):
    """Latent-first orchestration layer over vendored `VLAFlowMatching`.

    Vendored-preserved path:
    - If the selected expert is the flow passthrough wrapper, we delegate to
      vendored `super().forward` / `super().sample_actions`.

    Project logic:
    - Extract shared VLM-conditioned routing latents.
    - Route in both training and inference.
    - Dispatch to a router-selected expert from the router registry.
    """

    def __init__(
        self,
        config,
        rtc_processor=None,
        router: ActionExpertRouter | None = None,
    ):
        super().__init__(config, rtc_processor=rtc_processor)
        self.router = router if router is not None else FlowOnlyRouter()

        if len(self.router.available_experts) == 0:
            raise ValueError("Router registry must contain at least one expert.")

        self.last_router_decision: dict[str, object] = {
            "router_choice": self.router.available_experts[0],
            "router_confidence": 1.0,
            "router_metadata": {},
        }

    def _set_router_decision(self, *, expert_id: str, confidence: float, metadata: dict[str, object]) -> None:
        self.last_router_decision = {
            "router_choice": expert_id,
            "router_confidence": float(confidence),
            "router_metadata": dict(metadata),
        }

    def _extract_training_latents(
        self,
        *,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise,
        time,
    ) -> ExpertTrainingBatch:
        """Extract training payload from vendored forward pipeline.

        Vendored-copied/adapted logic:
        - x_t interpolation and velocity target construction.
        - Prefix/suffix embedding and transformer forward wiring.

        Project logic:
        - Expose a shared routing latent contract consumed by routers/experts.
        """

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        velocity_target = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        prefix_out, suffix_out = outputs_embeds
        if prefix_out is None or suffix_out is None:
            raise RuntimeError("Expected both prefix and suffix outputs from vlm_with_expert.forward.")

        routing_latents = RoutingLatents(
            context_latent=self._masked_mean_pool(prefix_out, prefix_pad_masks),
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=None,
        )

        return ExpertTrainingBatch(
            routing=routing_latents,
            suffix_out=suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32),
            actions=actions,
            velocity_target=velocity_target,
        )

    def _extract_sampling_latents(
        self,
        *,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) -> tuple[RoutingLatents, Tensor]:
        """Extract inference routing payload from vendored sample pipeline.

        Vendored-copied/adapted logic:
        - Prefix embedding and KV-cache construction for denoising rollout.
        """

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        context_latent, past_key_values = self._compute_prefix_context_and_cache(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_2d_masks,
            prefix_position_ids,
        )

        routing_latents = RoutingLatents(
            context_latent=context_latent,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
        )
        return routing_latents, noise

    def _masked_mean_pool(self, tokens: Tensor, pad_mask: Tensor) -> Tensor:
        mask = pad_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return (tokens * mask).sum(dim=1) / denom

    def _route(self, routing_latents: RoutingLatents):
        decision = self.router.route(routing_latents=routing_latents)
        self.router.get_expert(decision.expert_id)
        self._set_router_decision(
            expert_id=decision.expert_id,
            confidence=decision.confidence,
            metadata=decision.metadata,
        )
        return decision

    def _compute_prefix_context_and_cache(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_2d_masks: Tensor,
        prefix_position_ids: Tensor,
    ):
        """Compute prefix KV cache and pooled prefix latent for routing.

        Vendored-preserved call:
        - Transformer call shape/flags that fill the prefix KV cache.

        Project logic:
        - Mean-pool prefix outputs into `context_latent` for router input.
        """

        outputs_embeds, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        prefix_out = outputs_embeds[0]
        if prefix_out is None:
            raise RuntimeError("Expected prefix transformer output for routing context, got None.")

        context_latent = self._masked_mean_pool(prefix_out, prefix_pad_masks)
        return context_latent, past_key_values

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        latents = self._extract_training_latents(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=actions,
            noise=noise,
            time=time,
        )

        decision = self._route(latents.routing)
        expert = self.router.get_expert(decision.expert_id)

        # Strict passthrough path: delegate to vendored implementation.
        if expert.use_vendor_passthrough():
            return super().forward(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                actions=actions,
                noise=noise,
                time=time,
            )

        return expert.training_loss(
            self,
            suffix_out=latents.suffix_out,
            routing_latents=latents.routing,
            actions=latents.actions,
            velocity_target=latents.velocity_target,
        )

    def denoise_step_hidden(
        self,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Expose suffix hidden states for non-vendored action heads.

        Vendored-copied/adapted logic:
        - Attention mask and position-id construction from denoise_step.

        Project logic:
        - Return hidden suffix states before any action projection head.
        """

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )

        suffix_out = outputs_embeds[1]
        if suffix_out is None:
            raise RuntimeError("Expected suffix output in denoise_step_hidden, got None.")
        return suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs,
    ) -> Tensor:
        routing_latents, sampled_noise = self._extract_sampling_latents(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
        )

        decision = self._route(routing_latents)
        expert = self.router.get_expert(decision.expert_id)

        # Strict passthrough path: delegate to vendored implementation.
        if expert.use_vendor_passthrough():
            return super().sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=sampled_noise,
                **kwargs,
            )

        return expert.sample_actions(
            self,
            routing_latents=routing_latents,
            initial_noise=sampled_noise,
            **kwargs,
        )


class MyPolicy(SmolVLAPolicy):
    """Policy wrapper preserving vendor lifecycle while swapping model internals."""

    def __init__(self, config, router: ActionExpertRouter | str | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self._router = self._build_router(router)
        self.model = RoutedVLAFlowMatching(config, rtc_processor=self.rtc_processor, router=self._router)

    def _build_router(self, router: ActionExpertRouter | str | None) -> ActionExpertRouter:
        if isinstance(router, ActionExpertRouter):
            return router

        if router is None or router == "flow_only":
            return FlowOnlyRouter()

        if router == "binning_only":
            hidden_size = self.model.vlm_with_expert.expert_hidden_size
            return BinningOnlyRouter(hidden_size=hidden_size, max_action_dim=self.config.max_action_dim)

        raise ValueError(f"Unknown router '{router}'. Expected one of: flow_only, binning_only.")

    def get_routing_metadata(self) -> dict[str, object]:
        return {
            "router_impl": self._router.__class__.__name__,
            "available_experts": self._router.available_experts,
        }


def build_mypolicy_flow_only() -> MyPolicy:
    return MyPolicy.from_pretrained(OFFICIAL_MODEL_ID, router="flow_only")


def build_mypolicy_binning_only() -> MyPolicy:
    return MyPolicy.from_pretrained(OFFICIAL_MODEL_ID, router="binning_only")
