"""Normalization layers used by AtomLLM."""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Bias-free RMSNorm with at least FP32 internal statistics."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if type(eps) not in {int, float} or not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_floating_point():
            raise TypeError("RMSNorm input must be floating point")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected last dimension {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        input_dtype = hidden_states.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in {torch.float16, torch.bfloat16}
            else input_dtype
        )
        values = hidden_states.to(compute_dtype)
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.to(compute_dtype)).to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
