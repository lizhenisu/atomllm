"""Complete AtomLLM decoder-only causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from atomllm.model.attention import KVCache
from atomllm.model.block import TransformerBlock
from atomllm.model.config import ModelConfig
from atomllm.model.normalization import RMSNorm


ModelCache = tuple[KVCache, ...]


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    past_key_values: ModelCache | None


class _TiedLMHead(nn.Module):
    """Linear projection that registers the embedding Parameter without copying it."""

    def __init__(self, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return functional.linear(hidden_states, self.weight)


class AtomLLM(nn.Module):
    """Pre-Norm decoder with tied embeddings and per-layer compact KV caches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.tokenizer.vocab_size
        self.hidden_size = config.dimensions.hidden_size
        self.num_layers = config.dimensions.num_layers
        self.max_sequence_length = config.dimensions.max_sequence_length
        self.pad_token_id = config.tokenizer.special_token_ids["pad"]

        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        nn.init.normal_(
            self.token_embeddings.weight,
            mean=0.0,
            std=config.components.initializer_std,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_index)
            for layer_index in range(self.num_layers)
        )
        self.final_norm = RMSNorm(
            self.hidden_size,
            eps=config.components.rms_norm_epsilon,
        )
        self.lm_head = _TiedLMHead(self.token_embeddings.weight)

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("input_ids must use int32 or int64 dtype")
        if input_ids.shape[1] <= 0:
            raise ValueError("input sequence length must be positive")
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size:
            raise ValueError(f"input_ids must be in [0, {self.vocab_size})")

    def _validate_cache(
        self,
        past_key_values: ModelCache | None,
    ) -> tuple[KVCache | None, ...]:
        if past_key_values is None:
            return (None,) * self.num_layers
        if not isinstance(past_key_values, tuple):
            raise TypeError("past_key_values must be a tuple")
        if len(past_key_values) != self.num_layers:
            raise ValueError(
                f"past_key_values must contain {self.num_layers} layer caches"
            )
        sequence_lengths = {cache.sequence_length for cache in past_key_values}
        if len(sequence_lengths) != 1:
            raise ValueError("all layer caches must have the same sequence length")
        return past_key_values

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must have the same [batch, sequence] shape")
        if labels.dtype != torch.int64:
            raise TypeError("labels must use int64 dtype")
        if labels.device != logits.device:
            raise ValueError("labels and logits must be on the same device")
        if labels.shape[1] < 2:
            raise ValueError("at least two tokens are required to calculate loss")
        valid_labels = labels == -100
        in_vocabulary = (labels >= 0) & (labels < self.vocab_size)
        if not torch.all(valid_labels | in_vocabulary).item():
            raise ValueError(f"labels must be -100 or in [0, {self.vocab_size})")

        effective_labels = labels.clone()
        effective_labels[effective_labels == self.pad_token_id] = -100
        if attention_mask is not None:
            if attention_mask.shape != labels.shape:
                raise ValueError(
                    "attention_mask must match labels when calculating loss"
                )
            effective_labels.masked_fill_(
                ~attention_mask.to(device=labels.device, dtype=torch.bool),
                -100,
            )
        shift_logits = logits[:, :-1].float().contiguous()
        shift_labels = effective_labels[:, 1:].contiguous()
        if not torch.any(shift_labels != -100).item():
            raise ValueError("loss has no valid next-token targets")
        return functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: ModelCache | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        self._validate_input_ids(input_ids)
        if labels is not None and past_key_values is not None:
            raise ValueError("labels with past_key_values are not supported")
        layer_caches = self._validate_cache(past_key_values)
        hidden_states = self.token_embeddings(input_ids)
        next_caches: list[KVCache] = []
        for layer, layer_cache in zip(
            self.layers,
            layer_caches,
            strict=True,
        ):
            hidden_states, next_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                if next_cache is None:
                    raise RuntimeError("layer did not return a requested KV cache")
                next_caches.append(next_cache)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = (
            self._loss(logits, labels, attention_mask) if labels is not None else None
        )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(next_caches) if use_cache else None,
        )
