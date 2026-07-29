"""Grouped-query causal self-attention with compact KV caching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from atomllm.model.config import ModelConfig
from atomllm.model.rotary import RotaryEmbedding


def _create_segment_block_mask(segment_ids: torch.Tensor) -> BlockMask:
    batch_size, sequence_length = segment_ids.shape

    def mask_mod(
        batch: torch.Tensor,
        _head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        query_segment = segment_ids[batch, query_index]
        return (
            (query_index >= key_index)
            & (query_segment > 0)
            & (query_segment == segment_ids[batch, key_index])
        )

    return create_block_mask(
        mask_mod,
        batch_size,
        None,
        sequence_length,
        sequence_length,
        device=segment_ids.device,
    )


_compiled_create_segment_block_mask = torch.compile(
    _create_segment_block_mask,
    fullgraph=True,
    dynamic=False,
)
_compiled_flex_attention = torch.compile(
    flex_attention,
    fullgraph=True,
    dynamic=False,
)


def build_segment_attention_mask(
    segment_ids: torch.Tensor,
    *,
    use_flex: bool = True,
) -> BlockMask | torch.Tensor:
    """Build causal attention that cannot cross packed-conversation boundaries."""
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape [batch, sequence]")
    if segment_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("segment_ids must use an integer dtype")
    if torch.any(segment_ids < 0).item():
        raise ValueError("segment_ids must be non-negative")
    if segment_ids.device.type == "cuda" and use_flex:
        return _compiled_create_segment_block_mask(segment_ids)

    query_segments = segment_ids[:, None, :, None]
    key_segments = segment_ids[:, None, None, :]
    sequence_length = segment_ids.shape[1]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=segment_ids.device,
    ).tril()
    return causal[None, None] & (query_segments > 0) & (query_segments == key_segments)


@dataclass(frozen=True, slots=True)
class KVCache:
    """Rotated key and value tensors kept at the configured KV-head count."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return self.key.shape[-2]


class GroupedQueryAttention(nn.Module):
    """Bias-free GQA using explicit causal masks for training and cached decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dimensions = config.dimensions
        components = config.components
        self.hidden_size = dimensions.hidden_size
        self.num_attention_heads = dimensions.num_attention_heads
        self.num_key_value_heads = dimensions.num_key_value_heads
        self.head_dim = dimensions.head_dim
        self.max_sequence_length = dimensions.max_sequence_length
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = components.attention_dropout

        query_width = self.num_attention_heads * self.head_dim
        key_value_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            query_width,
            bias=components.use_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            key_value_width,
            bias=components.use_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            key_value_width,
            bias=components.use_bias,
        )
        self.o_proj = nn.Linear(
            query_width,
            self.hidden_size,
            bias=components.use_bias,
        )
        self.rotary = RotaryEmbedding(
            self.head_dim,
            theta=components.rope_theta,
            max_sequence_length=self.max_sequence_length,
        )
        self._initialize_weights(config)

    def _initialize_weights(self, config: ModelConfig) -> None:
        standard_deviation = config.components.initializer_std
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=standard_deviation)
        output_standard_deviation = standard_deviation
        if config.components.scale_residual_projections:
            output_standard_deviation /= math.sqrt(2 * config.dimensions.num_layers)
        nn.init.normal_(
            self.o_proj.weight,
            mean=0.0,
            std=output_standard_deviation,
        )

    def _shape_query(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        return (
            values.view(
                batch_size,
                sequence_length,
                self.num_attention_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _shape_key_value(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        return (
            values.view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _validate_cache(
        self,
        cache: KVCache,
        hidden_states: torch.Tensor,
    ) -> None:
        expected_prefix = (
            hidden_states.shape[0],
            self.num_key_value_heads,
        )
        for name, values in (("key", cache.key), ("value", cache.value)):
            if values.ndim != 4:
                raise ValueError(f"cached {name} must be four-dimensional")
            if values.shape[:2] != expected_prefix:
                raise ValueError(
                    f"cached {name} must have batch={expected_prefix[0]} and "
                    f"kv_heads={expected_prefix[1]}"
                )
            if values.shape[-1] != self.head_dim:
                raise ValueError(
                    f"cached {name} head dimension must be {self.head_dim}"
                )
            if values.device != hidden_states.device:
                raise ValueError(f"cached {name} must be on the input device")
            if values.dtype != hidden_states.dtype:
                raise ValueError(f"cached {name} must use the input dtype")
        if cache.key.shape != cache.value.shape:
            raise ValueError("cached key and value shapes must match")
        if cache.sequence_length <= 0:
            raise ValueError("cached sequence length must be positive")

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        total_sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask.ndim != 2 or attention_mask.shape != (
            batch_size,
            total_sequence_length,
        ):
            raise ValueError(
                "attention_mask must have shape [batch, total_sequence_length]"
            )
        if attention_mask.dtype != torch.bool and attention_mask.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("attention_mask must use boolean or integer dtype")
        mask = attention_mask.to(device=device)
        if mask.dtype != torch.bool:
            if not torch.all((mask == 0) | (mask == 1)).item():
                raise ValueError("integer attention_mask values must be 0 or 1")
            mask = mask.to(torch.bool)
        return mask

    def _attention_mask(
        self,
        batch_size: int,
        query_length: int,
        past_length: int,
        device: torch.device,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        total_length = past_length + query_length
        query_positions = (
            torch.arange(query_length, device=device) + past_length
        ).unsqueeze(-1)
        key_positions = torch.arange(total_length, device=device).unsqueeze(0)
        allowed = (key_positions <= query_positions).view(
            1,
            1,
            query_length,
            total_length,
        )
        if attention_mask is None:
            return allowed
        valid_tokens = self._validate_attention_mask(
            attention_mask,
            batch_size,
            total_length,
            device,
        )
        allowed = allowed & valid_tokens[:, None, None, :]
        query_valid = valid_tokens[:, past_length:total_length]
        return allowed & query_valid[:, None, :, None]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        segment_attention_mask: BlockMask | torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden_size]"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")
        if hidden_states.shape[1] <= 0:
            raise ValueError("query sequence length must be positive")

        past_length = 0
        if past_key_value is not None:
            self._validate_cache(past_key_value, hidden_states)
            past_length = past_key_value.sequence_length
        query_length = hidden_states.shape[1]
        total_length = past_length + query_length
        if total_length > self.max_sequence_length:
            raise ValueError(
                f"attention sequence length {total_length} exceeds "
                f"max_sequence_length {self.max_sequence_length}"
            )

        query = self._shape_query(self.q_proj(hidden_states))
        key = self._shape_key_value(self.k_proj(hidden_states))
        value = self._shape_key_value(self.v_proj(hidden_states))
        query, key = self.rotary(query, key, offset=past_length)

        if past_key_value is not None:
            key = torch.cat((past_key_value.key, key), dim=-2)
            value = torch.cat((past_key_value.value, value), dim=-2)
        next_cache = KVCache(key=key, value=value) if use_cache else None
        if segment_attention_mask is not None:
            if attention_mask is not None:
                raise ValueError(
                    "segment_attention_mask and attention_mask are mutually exclusive"
                )
            if past_key_value is not None or use_cache:
                raise ValueError("segment attention is only supported without KV cache")
            if self.training and self.attention_dropout:
                raise ValueError("segment attention requires zero attention dropout")
            if isinstance(segment_attention_mask, BlockMask):
                attention_output = _compiled_flex_attention(
                    query,
                    key,
                    value,
                    block_mask=segment_attention_mask,
                    enable_gqa=True,
                )
            else:
                attention_output = functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=segment_attention_mask,
                    dropout_p=0.0,
                    enable_gqa=True,
                )
            attention_output = (
                attention_output.transpose(1, 2)
                .contiguous()
                .view(hidden_states.shape[0], query_length, self.hidden_size)
            )
            return self.o_proj(attention_output), next_cache
        # The common pre-training path has no padding and no KV cache. Let
        # SDPA express causality directly so CUDA can select a fused kernel and
        # avoid materializing a [T, T] boolean mask for every block.
        direct_causal = past_key_value is None and attention_mask is None
        allowed = None
        if not direct_causal:
            allowed = self._attention_mask(
                hidden_states.shape[0],
                query_length,
                past_length,
                hidden_states.device,
                attention_mask,
            )
        attention_output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=direct_causal,
            enable_gqa=True,
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], query_length, self.hidden_size)
        )
        return self.o_proj(attention_output), next_cache
