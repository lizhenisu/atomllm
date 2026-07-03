"""Rotary position embedding used by AtomLLM attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Full-head Llama-style RoPE for query and key tensors."""

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float = 10_000.0,
        max_sequence_length: int = 8_192,
    ) -> None:
        super().__init__()
        if type(head_dim) is not int or head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if (
            type(theta) not in {int, float}
            or not math.isfinite(float(theta))
            or theta <= 0
        ):
            raise ValueError("theta must be a positive finite number")
        if type(max_sequence_length) is not int or max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be a positive integer")
        self.head_dim = head_dim
        self.theta = float(theta)
        self.max_sequence_length = max_sequence_length
        frequencies = 1.0 / (
            self.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", frequencies, persistent=False)

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[int, int]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError(
                "query and key must have shape [batch, heads, sequence, dim]"
            )
        if query.shape[0] != key.shape[0] or query.shape[-2] != key.shape[-2]:
            raise ValueError("query and key batch and sequence dimensions must match")
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError(f"query and key head dimension must be {self.head_dim}")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        if query.dtype != key.dtype:
            raise ValueError("query and key must have the same dtype")
        if not query.is_floating_point() or not key.is_floating_point():
            raise TypeError("query and key must be floating point")
        batch_size = query.shape[0]
        sequence_length = query.shape[-2]
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        return batch_size, sequence_length

    def _positions(
        self,
        query: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        position_ids: torch.Tensor | None,
        offset: int,
    ) -> torch.Tensor:
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if position_ids is not None and offset != 0:
            raise ValueError("offset must be zero when position_ids are provided")
        if position_ids is None:
            positions = torch.arange(
                offset,
                offset + sequence_length,
                device=query.device,
                dtype=torch.long,
            )
        else:
            if position_ids.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("position_ids must use an integer dtype")
            if position_ids.ndim == 1:
                if position_ids.shape[0] != sequence_length:
                    raise ValueError(
                        "1D position_ids length must match sequence length"
                    )
            elif position_ids.ndim == 2:
                if position_ids.shape != (batch_size, sequence_length):
                    raise ValueError(
                        "2D position_ids must have shape [batch, sequence]"
                    )
            else:
                raise ValueError("position_ids must be 1D or 2D")
            positions = position_ids.to(device=query.device, dtype=torch.long)
        if positions.min().item() < 0:
            raise ValueError("position_ids must be non-negative")
        if positions.max().item() >= self.max_sequence_length:
            raise ValueError(
                "position exceeds configured max_sequence_length "
                f"{self.max_sequence_length}"
            )
        return positions

    def _cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.to(
            positions.device
        )
        embeddings = torch.cat((angles, angles), dim=-1)
        cos = embeddings.cos()
        sin = embeddings.sin()
        if positions.ndim == 1:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return cos.to(dtype), sin.to(dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = self._validate_inputs(query, key)
        positions = self._positions(
            query,
            batch_size,
            sequence_length,
            position_ids,
            offset,
        )
        cos, sin = self._cos_sin(positions, torch.float32)
        query_float = query.to(torch.float32)
        key_float = key.to(torch.float32)
        rotated_query = (query_float * cos + self._rotate_half(query_float) * sin).to(
            query.dtype
        )
        rotated_key = (key_float * cos + self._rotate_half(key_float) * sin).to(
            key.dtype
        )
        return rotated_query, rotated_key

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"max_sequence_length={self.max_sequence_length}"
        )
