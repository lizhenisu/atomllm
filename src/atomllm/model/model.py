"""Complete AtomLLM decoder-only causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint

from atomllm.model.attention import KVCache
from atomllm.model.block import TransformerBlock
from atomllm.model.config import ModelConfig
from atomllm.model.normalization import RMSNorm


ModelCache = tuple[KVCache, ...]


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    logits: torch.Tensor | None
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
        values_are_valid = torch.all((input_ids >= 0) & (input_ids < self.vocab_size))
        if input_ids.device.type == "cuda":
            torch._assert_async(
                values_are_valid,
                f"input_ids must be in [0, {self.vocab_size})",
            )
        elif not values_are_valid.item():
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
        labels_are_valid = torch.all(valid_labels | in_vocabulary)
        if labels.device.type == "cuda":
            torch._assert_async(
                labels_are_valid,
                f"labels must be -100 or in [0, {self.vocab_size})",
            )
        elif not labels_are_valid.item():
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
        has_targets = torch.any(shift_labels != -100)
        if labels.device.type == "cuda":
            torch._assert_async(has_targets, "loss has no valid next-token targets")
        elif not has_targets.item():
            raise ValueError("loss has no valid next-token targets")
        return functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _effective_labels(
        self,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        device: torch.device,
        sequence_length: int,
    ) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != sequence_length:
            raise ValueError("labels must have the same [batch, sequence] shape")
        if labels.dtype != torch.int64:
            raise TypeError("labels must use int64 dtype")
        if labels.device != device:
            raise ValueError("labels and hidden states must be on the same device")
        if sequence_length < 2:
            raise ValueError("at least two tokens are required to calculate loss")
        ignored = labels == -100
        in_vocabulary = (labels >= 0) & (labels < self.vocab_size)
        labels_are_valid = torch.all(ignored | in_vocabulary)
        if labels.device.type == "cuda":
            torch._assert_async(
                labels_are_valid,
                f"labels must be -100 or in [0, {self.vocab_size})",
            )
        elif not labels_are_valid.item():
            raise ValueError(f"labels must be -100 or in [0, {self.vocab_size})")
        effective = labels.clone()
        effective[effective == self.pad_token_id] = -100
        if attention_mask is not None:
            if attention_mask.shape != labels.shape:
                raise ValueError(
                    "attention_mask must match labels when calculating loss"
                )
            effective.masked_fill_(
                ~attention_mask.to(device=device, dtype=torch.bool),
                -100,
            )
        has_targets = torch.any(effective[:, 1:] != -100)
        if labels.device.type == "cuda":
            torch._assert_async(has_targets, "loss has no valid next-token targets")
        elif not has_targets.item():
            raise ValueError("loss has no valid next-token targets")
        return effective

    def _chunked_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        chunk_size: int,
    ) -> torch.Tensor:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("loss_chunk_size must be a positive integer")
        effective = self._effective_labels(
            labels,
            attention_mask,
            device=hidden_states.device,
            sequence_length=hidden_states.shape[1],
        )
        shifted_hidden = hidden_states[:, :-1]
        shifted_labels = effective[:, 1:]
        valid_count = (shifted_labels != -100).sum()
        loss_sum = hidden_states.new_zeros((), dtype=torch.float32)

        def chunk_loss(states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            logits = functional.linear(states, self.lm_head.weight).float()
            return functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        for start in range(0, shifted_hidden.shape[1], chunk_size):
            states = shifted_hidden[:, start : start + chunk_size]
            targets = shifted_labels[:, start : start + chunk_size]
            if self.training and torch.is_grad_enabled():
                partial = checkpoint(
                    chunk_loss,
                    states,
                    targets,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                partial = chunk_loss(states, targets)
            loss_sum = loss_sum + partial
        return loss_sum / valid_count

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: ModelCache | None = None,
        use_cache: bool = False,
        loss_chunk_size: int | None = None,
        gradient_checkpointing: bool = False,
        checkpoint_segment_layers: int = 1,
        checkpoint_interval_segments: int = 1,
    ) -> CausalLMOutput:
        self._validate_input_ids(input_ids)
        if labels is not None and past_key_values is not None:
            raise ValueError("labels with past_key_values are not supported")
        if gradient_checkpointing and (not self.training or use_cache):
            raise ValueError(
                "gradient checkpointing requires training mode without KV cache"
            )
        if type(checkpoint_segment_layers) is not int or checkpoint_segment_layers <= 0:
            raise ValueError("checkpoint_segment_layers must be a positive integer")
        if (
            type(checkpoint_interval_segments) is not int
            or checkpoint_interval_segments <= 0
        ):
            raise ValueError("checkpoint_interval_segments must be a positive integer")
        layer_caches = self._validate_cache(past_key_values)
        hidden_states = self.token_embeddings(input_ids)
        next_caches: list[KVCache] = []
        if gradient_checkpointing:
            for segment_index, start in enumerate(
                range(0, self.num_layers, checkpoint_segment_layers)
            ):
                segment = tuple(self.layers[start : start + checkpoint_segment_layers])

                def segment_forward(
                    states: torch.Tensor,
                    current_segment: tuple[TransformerBlock, ...] = segment,
                ) -> torch.Tensor:
                    for current_layer in current_segment:
                        states = current_layer(
                            states,
                            attention_mask=attention_mask,
                            past_key_value=None,
                            use_cache=False,
                        )[0]
                    return states

                checkpoint_segment = (
                    checkpoint_interval_segments == 1
                    or (segment_index + 1) % checkpoint_interval_segments != 0
                )
                if checkpoint_segment:
                    hidden_states = checkpoint(
                        segment_forward,
                        hidden_states,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    hidden_states = segment_forward(hidden_states)
        else:
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
        if labels is not None and loss_chunk_size is not None:
            logits = None
            loss = self._chunked_loss(
                hidden_states,
                labels,
                attention_mask,
                loss_chunk_size,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss = (
                self._loss(logits, labels, attention_mask)
                if labels is not None
                else None
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(next_caches) if use_cache else None,
        )
