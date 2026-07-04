"""SwiGLU feed-forward network used by AtomLLM blocks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

from atomllm.model.config import ModelConfig


class SwiGLU(nn.Module):
    """Bias-free gated MLP with scaled residual projection initialization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dimensions = config.dimensions
        components = config.components
        self.hidden_size = dimensions.hidden_size
        self.ffn_hidden_size = dimensions.ffn_hidden_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.ffn_hidden_size,
            bias=components.use_bias,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.ffn_hidden_size,
            bias=components.use_bias,
        )
        self.down_proj = nn.Linear(
            self.ffn_hidden_size,
            self.hidden_size,
            bias=components.use_bias,
        )
        self._initialize_weights(config)

    def _initialize_weights(self, config: ModelConfig) -> None:
        standard_deviation = config.components.initializer_std
        for projection in (self.gate_proj, self.up_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=standard_deviation)
        output_standard_deviation = standard_deviation
        if config.components.scale_residual_projections:
            output_standard_deviation /= math.sqrt(2 * config.dimensions.num_layers)
        nn.init.normal_(
            self.down_proj.weight,
            mean=0.0,
            std=output_standard_deviation,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden_size]"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")
        gate = functional.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)
