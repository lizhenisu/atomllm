"""Minimal greedy generation with optional multi-layer KV caching."""

from __future__ import annotations

import torch

from atomllm.model.model import AtomLLM, ModelCache


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
