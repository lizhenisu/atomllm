"""Pre-Norm Transformer block used by AtomLLM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from atomllm.model.attention import GroupedQueryAttention, KVCache
from atomllm.model.config import ModelConfig
from atomllm.model.mlp import SwiGLU
from atomllm.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Attention and SwiGLU residual branches with independent RMSNorms."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        if (
            type(layer_index) is not int
            or layer_index < 0
            or layer_index >= config.dimensions.num_layers
        ):
            raise ValueError(
                f"layer_index must be in [0, {config.dimensions.num_layers})"
            )
        self.layer_index = layer_index
        self.hidden_size = config.dimensions.hidden_size
        self.attention_norm = RMSNorm(
            self.hidden_size,
            eps=config.components.rms_norm_epsilon,
        )
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(
            self.hidden_size,
            eps=config.components.rms_norm_epsilon,
        )
        self.mlp = SwiGLU(config)
        # Pre-training uses zero dropout. Avoid dispatching a no-op dropout
        # operator twice per layer while retaining configurable dropout for
        # smaller-data post-training model configs.
        self.residual_dropout = (
            nn.Identity()
            if config.components.residual_dropout == 0.0
            else nn.Dropout(config.components.residual_dropout)
        )

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
        attention_output, next_cache = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            segment_attention_mask=segment_attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.residual_dropout(attention_output)
        mlp_output = self.mlp(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + self.residual_dropout(mlp_output)
        return hidden_states, next_cache
