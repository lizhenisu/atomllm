from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from atomllm.model.attention import KVCache
from atomllm.model.block import TransformerBlock
from atomllm.model.config import load_model_config
from atomllm.model.mlp import SwiGLU


CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")


def small_model_config():
    config = load_model_config(CONFIG_PATH)
    return replace(
        config,
        dimensions=replace(
            config.dimensions,
            max_sequence_length=32,
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            ffn_hidden_size=64,
        ),
    )


def make_block() -> TransformerBlock:
    torch.manual_seed(23)
    return TransformerBlock(small_model_config(), layer_index=0).eval()


def test_swiglu_matches_explicit_reference() -> None:
    torch.manual_seed(29)
    mlp = SwiGLU(small_model_config())
    hidden_states = torch.randn(2, 3, 32)

    actual = mlp(hidden_states)
    expected = mlp.down_proj(
        torch.nn.functional.silu(mlp.gate_proj(hidden_states))
        * mlp.up_proj(hidden_states)
    )

    torch.testing.assert_close(actual, expected)


def test_swiglu_parameter_count_and_scaled_down_initialization() -> None:
    config = small_model_config()
    torch.manual_seed(31)
    mlp = SwiGLU(config)

    parameter_count = sum(parameter.numel() for parameter in mlp.parameters())
    expected_down_std = (
        config.components.initializer_std / (2 * config.dimensions.num_layers) ** 0.5
    )

    assert parameter_count == 6_144
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None
    assert mlp.down_proj.weight.std().item() == pytest.approx(
        expected_down_std,
        rel=0.08,
    )


def test_transformer_block_parameter_count_matches_contract() -> None:
    block = make_block()

    parameter_count = sum(parameter.numel() for parameter in block.parameters())

    assert parameter_count == 9_280


class _ScaledAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        return hidden_states * 2, None


class _ScaledMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 3


def test_transformer_block_uses_sequential_prenorm_residual_order() -> None:
    block = make_block()
    block.attention_norm = nn.Identity()
    block.attention = _ScaledAttention()
    block.ffn_norm = nn.Identity()
    block.mlp = _ScaledMLP()
    hidden_states = torch.randn(2, 4, 32)

    output, cache = block(hidden_states)

    torch.testing.assert_close(output, hidden_states * 12)
    assert cache is None


def test_transformer_block_cache_matches_full_sequence() -> None:
    block = make_block()
    hidden_states = torch.randn(1, 7, 32)

    full_output, _ = block(hidden_states)
    pieces = []
    cache = None
    for index in range(hidden_states.shape[1]):
        output, cache = block(
            hidden_states[:, index : index + 1],
            attention_mask=torch.ones(1, index + 1, dtype=torch.bool),
            past_key_value=cache,
            use_cache=True,
        )
        pieces.append(output)

    torch.testing.assert_close(
        torch.cat(pieces, dim=1),
        full_output,
        rtol=1e-5,
        atol=1e-5,
    )
    assert isinstance(cache, KVCache)
    assert cache.key.shape == (1, 2, 7, 8)


def test_transformer_block_backward_has_finite_gradients() -> None:
    block = make_block().train()
    hidden_states = torch.randn(2, 5, 32, requires_grad=True)

    output, _ = block(hidden_states)
    output.square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )


def test_transformer_block_bfloat16_cpu() -> None:
    block = make_block().to(dtype=torch.bfloat16).train()
    hidden_states = torch.randn(
        1,
        4,
        32,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output, cache = block(hidden_states, use_cache=True)
    output.float().mean().backward()

    assert output.dtype == torch.bfloat16
    assert cache is not None and cache.key.dtype == torch.bfloat16
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_transformer_block_rejects_invalid_layer_index() -> None:
    config = small_model_config()

    with pytest.raises(ValueError, match="layer_index"):
        TransformerBlock(config, layer_index=-1)
    with pytest.raises(ValueError, match="layer_index"):
        TransformerBlock(config, layer_index=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_transformer_block_cuda_bfloat16_cache_smoke() -> None:
    block = make_block().to(device="cuda", dtype=torch.bfloat16).train()
    hidden_states = torch.randn(
        1,
        6,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    first, cache = block(hidden_states[:, :4], use_cache=True)
    second, cache = block(
        hidden_states[:, 4:],
        attention_mask=torch.ones(1, 6, device="cuda", dtype=torch.bool),
        past_key_value=cache,
        use_cache=True,
    )
    (first.float().mean() + second.float().mean()).backward()

    assert cache is not None and cache.key.shape == (1, 2, 6, 8)
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
