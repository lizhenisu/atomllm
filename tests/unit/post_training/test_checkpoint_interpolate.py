import pytest
import torch

from atomllm.post_training.checkpoint_interpolate import (
    CheckpointInterpolationError,
    _advance_interpolation,
)


def test_advance_interpolation_reaches_requested_absolute_alpha() -> None:
    current = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        current.weight.fill_(0.0)
        target.weight.fill_(8.0)

    _advance_interpolation(current, target, current_alpha=0.0, target_alpha=0.25)
    assert torch.equal(current.weight, torch.full_like(current.weight, 2.0))
    _advance_interpolation(current, target, current_alpha=0.25, target_alpha=0.75)
    assert torch.equal(current.weight, torch.full_like(current.weight, 6.0))


def test_advance_interpolation_rejects_invalid_alpha_order() -> None:
    model = torch.nn.Linear(1, 1)
    with pytest.raises(CheckpointInterpolationError, match="alphas"):
        _advance_interpolation(model, model, current_alpha=0.75, target_alpha=0.5)
