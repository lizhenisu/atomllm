from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.model.attention import GroupedQueryAttention, KVCache
from atomllm.model.config import load_model_config


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


def make_attention() -> GroupedQueryAttention:
    torch.manual_seed(17)
    return GroupedQueryAttention(small_model_config()).eval()


def test_gqa_output_and_compact_cache_shapes() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 5, 32)

    output, cache = attention(hidden_states, use_cache=True)

    assert output.shape == (2, 5, 32)
    assert cache is not None
    assert cache.key.shape == (2, 2, 5, 8)
    assert cache.value.shape == (2, 2, 5, 8)
    assert cache.sequence_length == 5


def test_gqa_parameter_count_matches_projection_formula() -> None:
    attention = make_attention()

    parameter_count = sum(parameter.numel() for parameter in attention.parameters())

    assert parameter_count == 3_072
    assert all(parameter.ndim == 2 for parameter in attention.parameters())


def test_causal_attention_does_not_leak_future_tokens() -> None:
    attention = make_attention()
    original = torch.randn(1, 6, 32)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10

    original_output, _ = attention(original)
    changed_output, _ = attention(changed)

    torch.testing.assert_close(
        original_output[:, :4],
        changed_output[:, :4],
        rtol=1e-5,
        atol=1e-5,
    )


def test_direct_causal_sdpa_matches_explicit_all_valid_mask() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 7, 32)

    direct, _ = attention(hidden_states)
    explicit, _ = attention(
        hidden_states,
        attention_mask=torch.ones(2, 7, dtype=torch.bool),
    )

    torch.testing.assert_close(direct, explicit, rtol=1e-5, atol=1e-6)


def test_padding_mask_blocks_masked_keys_and_zeros_masked_queries() -> None:
    attention = make_attention()
    original = torch.randn(1, 4, 32)
    changed = original.clone()
    changed[:, 1] = torch.randn_like(changed[:, 1]) * 100
    mask = torch.tensor([[1, 0, 1, 1]])

    original_output, _ = attention(original, attention_mask=mask)
    changed_output, _ = attention(changed, attention_mask=mask)

    torch.testing.assert_close(
        original_output[:, 2:],
        changed_output[:, 2:],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        original_output[:, 1],
        torch.zeros_like(original_output[:, 1]),
    )


def test_incremental_cache_matches_full_sequence() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 7, 32)

    full_output, _ = attention(hidden_states)
    cache = None
    pieces = []
    for index in range(hidden_states.shape[1]):
        piece, cache = attention(
            hidden_states[:, index : index + 1],
            attention_mask=torch.ones(2, index + 1, dtype=torch.bool),
            past_key_value=cache,
            use_cache=True,
        )
        pieces.append(piece)
    incremental_output = torch.cat(pieces, dim=1)

    torch.testing.assert_close(
        incremental_output,
        full_output,
        rtol=1e-5,
        atol=1e-5,
    )
    assert cache is not None
    assert cache.key.shape == (2, 2, 7, 8)


def test_chunked_cache_matches_full_sequence() -> None:
    attention = make_attention()
    hidden_states = torch.randn(1, 8, 32)

    full_output, _ = attention(hidden_states)
    first, cache = attention(hidden_states[:, :3], use_cache=True)
    second, cache = attention(
        hidden_states[:, 3:],
        attention_mask=torch.ones(1, 8, dtype=torch.bool),
        past_key_value=cache,
        use_cache=True,
    )

    torch.testing.assert_close(
        torch.cat((first, second), dim=1),
        full_output,
        rtol=1e-5,
        atol=1e-5,
    )
    assert cache is not None and cache.sequence_length == 8


def test_gqa_backward_has_finite_gradients() -> None:
    attention = make_attention().train()
    hidden_states = torch.randn(2, 5, 32, requires_grad=True)

    output, _ = attention(hidden_states)
    output.square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )


def test_rejects_invalid_attention_masks() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="must have shape"):
        attention(hidden_states, attention_mask=torch.ones(2, 3))
    with pytest.raises(TypeError, match="boolean or integer"):
        attention(hidden_states, attention_mask=torch.ones(2, 4) * 0.5)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        attention(
            hidden_states,
            attention_mask=torch.full((2, 4), 2, dtype=torch.long),
        )


def test_rejects_invalid_or_oversized_cache() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 2, 32)
    wrong_heads = KVCache(
        key=torch.randn(2, 4, 3, 8),
        value=torch.randn(2, 4, 3, 8),
    )

    with pytest.raises(ValueError, match="kv_heads=2"):
        attention(hidden_states, past_key_value=wrong_heads)

    oversized = KVCache(
        key=torch.randn(2, 2, 31, 8),
        value=torch.randn(2, 2, 31, 8),
    )
    with pytest.raises(ValueError, match="exceeds"):
        attention(hidden_states, past_key_value=oversized)


def test_bfloat16_cpu_forward_and_backward() -> None:
    attention = make_attention().to(dtype=torch.bfloat16).train()
    hidden_states = torch.randn(
        1,
        4,
        32,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output, cache = attention(hidden_states, use_cache=True)
    output.float().mean().backward()

    assert output.dtype == torch.bfloat16
    assert cache is not None and cache.key.dtype == torch.bfloat16
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bfloat16_cuda_cache_smoke() -> None:
    attention = make_attention().to(device="cuda", dtype=torch.bfloat16).eval()
    hidden_states = torch.randn(
        1,
        6,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    first, cache = attention(hidden_states[:, :4], use_cache=True)
    second, cache = attention(
        hidden_states[:, 4:],
        attention_mask=torch.ones(1, 6, device="cuda", dtype=torch.bool),
        past_key_value=cache,
        use_cache=True,
    )
    (first.float().mean() + second.float().mean()).backward()

    assert cache is not None
    assert cache.key.shape == (1, 2, 6, 8)
    assert cache.key.dtype == torch.bfloat16
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
