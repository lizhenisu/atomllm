from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.inference.generation import greedy_generate
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM


CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")


def tiny_model() -> AtomLLM:
    config = load_model_config(CONFIG_PATH)
    config = replace(
        config,
        tokenizer=replace(config.tokenizer, vocab_size=64),
        dimensions=replace(
            config.dimensions,
            max_sequence_length=16,
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            ffn_hidden_size=32,
        ),
        expected_parameter_count=5_712,
    )
    torch.manual_seed(73)
    return AtomLLM(config).eval()


def test_cached_and_uncached_greedy_generation_are_identical() -> None:
    model = tiny_model()
    prompt = torch.tensor([[2, 17, 18, 19]])

    without_cache = greedy_generate(
        model,
        prompt,
        max_new_tokens=6,
        use_cache=False,
    )
    with_cache = greedy_generate(
        model,
        prompt,
        max_new_tokens=6,
        use_cache=True,
    )

    assert without_cache.shape == (1, 10)
    assert torch.equal(with_cache, without_cache)


def test_generation_validates_length_and_restores_training_mode() -> None:
    model = tiny_model().train()
    prompt = torch.tensor([[2, 17, 18, 19]])

    generated = greedy_generate(
        model,
        prompt,
        max_new_tokens=1,
        use_cache=True,
    )

    assert generated.shape == (1, 5)
    assert model.training
    with pytest.raises(ValueError, match="positive integer"):
        greedy_generate(model, prompt, max_new_tokens=0, use_cache=True)
    with pytest.raises(ValueError, match="exceed"):
        greedy_generate(model, prompt, max_new_tokens=13, use_cache=True)
