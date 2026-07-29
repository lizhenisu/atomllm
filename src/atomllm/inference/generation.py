"""Minimal greedy generation with optional multi-layer KV caching."""

from __future__ import annotations

import torch

from atomllm.model.model import AtomLLM, ModelCache


def _apply_repetition_controls(
    logits: torch.Tensor,
    continuation: torch.Tensor,
    *,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    """Penalize repeated continuation tokens and ban repeated n-grams."""
    controlled = logits.clone()
    if continuation.numel() and repetition_penalty != 1.0:
        token_ids = continuation[0].unique()
        scores = controlled[0, token_ids]
        controlled[0, token_ids] = torch.where(
            scores < 0,
            scores * repetition_penalty,
            scores / repetition_penalty,
        )
    length = continuation.shape[1]
    if no_repeat_ngram_size == 1 and length:
        controlled[0, continuation[0].unique()] = float("-inf")
    elif no_repeat_ngram_size > 1 and length >= no_repeat_ngram_size - 1:
        prefix_length = no_repeat_ngram_size - 1
        prefix = continuation[0, -prefix_length:].tolist()
        banned: list[int] = []
        for start in range(length - no_repeat_ngram_size + 1):
            if continuation[0, start : start + prefix_length].tolist() == prefix:
                banned.append(int(continuation[0, start + prefix_length]))
        if banned:
            controlled[0, banned] = float("-inf")
    return controlled


@torch.inference_mode()
def greedy_generate(
    model: AtomLLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    use_cache: bool,
) -> torch.Tensor:
    """Generate a fixed number of tokens for an unpadded prompt."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    model._validate_input_ids(input_ids)
    if input_ids.shape[1] + max_new_tokens > model.max_sequence_length:
        raise ValueError("prompt and generated tokens exceed max_sequence_length")

    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    cache: ModelCache | None = None
    try:
        for _ in range(max_new_tokens):
            model_input = generated[:, -1:] if cache is not None else generated
            attention_mask = torch.ones(
                generated.shape,
                dtype=torch.bool,
                device=generated.device,
            )
            output = model(
                model_input,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=use_cache,
            )
            next_token = output.logits[:, -1:].argmax(dim=-1)
            generated = torch.cat((generated, next_token), dim=1)
            cache = output.past_key_values
    finally:
        model.train(was_training)
    return generated


@torch.inference_mode()
def sample_generate(
    model: AtomLLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_token_ids: frozenset[int] = frozenset(),
    seed: int | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Sample one unpadded sequence with KV cache and optional stop tokens."""
    if input_ids.shape[0] != 1:
        raise ValueError("sample generation currently requires batch size one")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1")
    if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be a non-negative integer")
    model._validate_input_ids(input_ids)
    if input_ids.shape[1] + max_new_tokens > model.max_sequence_length:
        raise ValueError("prompt and generated tokens exceed max_sequence_length")

    generator = torch.Generator(device=input_ids.device)
    if seed is not None:
        generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    prompt_length = input_ids.shape[1]
    cache: ModelCache | None = None
    try:
        for _ in range(max_new_tokens):
            model_input = generated[:, -1:] if cache is not None else generated
            attention_mask = torch.ones(
                generated.shape,
                dtype=torch.bool,
                device=generated.device,
            )
            output = model(
                model_input,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            logits = output.logits[:, -1].float() / temperature
            logits = _apply_repetition_controls(
                logits,
                generated[:, prompt_length:],
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            limit = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, limit, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
            remove = torch.cumsum(sorted_probabilities, dim=-1) > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            probabilities = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
            cache = output.past_key_values
            if int(next_token.item()) in stop_token_ids:
                break
    finally:
        model.train(was_training)
    return generated
