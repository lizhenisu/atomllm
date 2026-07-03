import pytest
import torch

from atomllm.model.normalization import RMSNorm
from atomllm.model.rotary import RotaryEmbedding


def test_rmsnorm_matches_reference_formula() -> None:
    torch.manual_seed(7)
    values = torch.randn(2, 3, 8, dtype=torch.float32)
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))

    actual = norm(values)
    expected = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * norm.weight

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_rmsnorm_preserves_bfloat16_and_has_finite_gradients() -> None:
    values = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    norm = RMSNorm(16).to(dtype=torch.bfloat16)

    output = norm(values)
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_rmsnorm_rejects_invalid_shape_and_dtype() -> None:
    norm = RMSNorm(8)

    with pytest.raises(ValueError, match="last dimension 8"):
        norm(torch.randn(2, 7))
    with pytest.raises(TypeError, match="floating point"):
        norm(torch.ones(2, 8, dtype=torch.long))


@pytest.mark.parametrize(
    ("hidden_size", "eps"),
    [(0, 1e-6), (8, 0.0), (8, float("inf"))],
)
def test_rmsnorm_rejects_invalid_configuration(
    hidden_size: int,
    eps: float,
) -> None:
    with pytest.raises(ValueError):
        RMSNorm(hidden_size, eps)


def test_rope_position_zero_is_identity_and_preserves_norm() -> None:
    torch.manual_seed(11)
    rope = RotaryEmbedding(8, max_sequence_length=16)
    query = torch.randn(2, 4, 4, 8)
    key = torch.randn(2, 2, 4, 8)

    rotated_query, rotated_key = rope(query, key)

    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
    torch.testing.assert_close(
        rotated_query.float().norm(dim=-1),
        query.float().norm(dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        rotated_key.float().norm(dim=-1),
        key.float().norm(dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rope_supports_gqa_shapes_and_position_offset() -> None:
    torch.manual_seed(13)
    rope = RotaryEmbedding(64, max_sequence_length=32)
    query = torch.randn(2, 16, 5, 64)
    key = torch.randn(2, 4, 5, 64)

    offset_query, offset_key = rope(query, key, offset=3)
    explicit_query, explicit_key = rope(
        query,
        key,
        position_ids=torch.arange(3, 8),
    )

    assert offset_query.shape == query.shape
    assert offset_key.shape == key.shape
    torch.testing.assert_close(offset_query, explicit_query)
    torch.testing.assert_close(offset_key, explicit_key)


def test_rope_supports_per_batch_position_ids() -> None:
    rope = RotaryEmbedding(8, max_sequence_length=16)
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 3, 8)
    positions = torch.tensor([[0, 1, 2], [3, 4, 5]])

    rotated_query, rotated_key = rope(query, key, position_ids=positions)
    first_query, first_key = rope(query[:1], key[:1], position_ids=positions[0])
    second_query, second_key = rope(query[1:], key[1:], position_ids=positions[1])

    torch.testing.assert_close(rotated_query[:1], first_query)
    torch.testing.assert_close(rotated_key[:1], first_key)
    torch.testing.assert_close(rotated_query[1:], second_query)
    torch.testing.assert_close(rotated_key[1:], second_key)


def test_rope_preserves_bfloat16_and_backpropagates() -> None:
    rope = RotaryEmbedding(16, max_sequence_length=16)
    query = torch.randn(2, 4, 6, 16, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(2, 2, 6, 16, dtype=torch.bfloat16, requires_grad=True)

    rotated_query, rotated_key = rope(query, key)
    (rotated_query.float().sum() + rotated_key.float().sum()).backward()

    assert rotated_query.dtype == torch.bfloat16
    assert rotated_key.dtype == torch.bfloat16
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


@pytest.mark.parametrize(
    "position_ids",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0, 16]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0, 1]]),
    ],
)
def test_rope_rejects_invalid_positions(position_ids: torch.Tensor) -> None:
    rope = RotaryEmbedding(8, max_sequence_length=16)
    query = torch.randn(2, 4, 2, 8)
    key = torch.randn(2, 2, 2, 8)

    with pytest.raises((TypeError, ValueError)):
        rope(query, key, position_ids=position_ids)


def test_rope_rejects_mismatched_query_and_key() -> None:
    rope = RotaryEmbedding(8)
    query = torch.randn(2, 4, 3, 8)

    with pytest.raises(ValueError, match="batch and sequence"):
        rope(query, torch.randn(1, 2, 3, 8))
    with pytest.raises(ValueError, match="head dimension"):
        rope(query, torch.randn(2, 2, 3, 6))
    with pytest.raises(ValueError, match="same dtype"):
        rope(query, torch.randn(2, 2, 3, 8, dtype=torch.bfloat16))


def test_rope_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="positive even"):
        RotaryEmbedding(7)
    with pytest.raises(ValueError, match="theta"):
        RotaryEmbedding(8, theta=0)
    with pytest.raises(ValueError, match="max_sequence_length"):
        RotaryEmbedding(8, max_sequence_length=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rmsnorm_and_rope_cuda_bfloat16_smoke() -> None:
    device = torch.device("cuda")
    norm = RMSNorm(64).to(device=device, dtype=torch.bfloat16)
    rope = RotaryEmbedding(64, max_sequence_length=32).to(device)
    hidden = torch.randn(
        2,
        8,
        64,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    query = norm(hidden).view(2, 8, 1, 64).transpose(1, 2)
    key = query.clone()

    rotated_query, rotated_key = rope(query, key, offset=4)
    (rotated_query.float().mean() + rotated_key.float().mean()).backward()

    assert rotated_query.dtype == torch.bfloat16
    assert rotated_query.device.type == "cuda"
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
