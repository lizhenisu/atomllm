from pathlib import Path

import pytest
import torch

from atomllm.model.config import calculate_parameter_count, load_model_config
from atomllm.training.micro_acceptance import _fixed_batch


CONFIG_PATH = Path("configs/model/atom-micro-4m.yaml")


def test_micro_model_config_uses_real_vocabulary_and_is_within_target_range() -> None:
    config = load_model_config(CONFIG_PATH)
    parameter_count = calculate_parameter_count(config).total

    assert config.tokenizer.vocab_size == 32_000
    assert parameter_count == 4_465_280
    assert 1_000_000 <= parameter_count <= 5_000_000


def test_fixed_batch_is_deterministic_and_avoids_reserved_tokens() -> None:
    first = _fixed_batch(32_000, 16, torch.device("cpu"))
    second = _fixed_batch(32_000, 16, torch.device("cpu"))

    assert torch.equal(first, second)
    assert first.shape == (1, 16)
    assert first.min().item() >= 15
    with pytest.raises(ValueError, match="at least 2"):
        _fixed_batch(32_000, 1, torch.device("cpu"))
