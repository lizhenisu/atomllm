from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from atomllm.inference.chat import encode_chat
from atomllm.inference.generation import (
    _apply_repetition_controls,
    greedy_generate,
    sample_generate,
)
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


def test_sample_generation_is_seeded_and_validates_sampling_parameters() -> None:
    model = tiny_model()
    prompt = torch.tensor([[2, 17, 18, 19]])

    first = sample_generate(model, prompt, max_new_tokens=4, seed=91)
    second = sample_generate(model, prompt, max_new_tokens=4, seed=91)

    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="temperature"):
        sample_generate(model, prompt, max_new_tokens=1, temperature=0)
    with pytest.raises(ValueError, match="top_p"):
        sample_generate(model, prompt, max_new_tokens=1, top_p=0)
    with pytest.raises(ValueError, match="repetition_penalty"):
        sample_generate(model, prompt, max_new_tokens=1, repetition_penalty=0.9)
    with pytest.raises(ValueError, match="no_repeat_ngram_size"):
        sample_generate(model, prompt, max_new_tokens=1, no_repeat_ngram_size=-1)


def test_repetition_controls_penalize_tokens_and_ban_repeated_ngrams() -> None:
    logits = torch.tensor([[0.0, 4.0, -2.0, 3.0, 2.0]])
    continuation = torch.tensor([[1, 2, 3, 1, 2]])

    controlled = _apply_repetition_controls(
        logits,
        continuation,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
    )

    assert controlled[0, 1].item() == 2.0
    assert controlled[0, 2].item() == -4.0
    assert controlled[0, 3].item() == float("-inf")
    assert controlled[0, 4].item() == 2.0


def test_chat_encoding_matches_frozen_sft_template() -> None:
    tokenizer = Tokenizer(WordLevel({"<unk>": 1, "hello": 20}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    tokens = encode_chat(tokenizer, [{"role": "user", "content": "hello"}])

    assert tokens == [2, 5, 20, 8, 6]
