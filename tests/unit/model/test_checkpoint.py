from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.model.checkpoint import (
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM


CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")


def tiny_model() -> AtomLLM:
    config = load_model_config(CONFIG_PATH)
    config = replace(
        config,
        name="atom-checkpoint-test",
        tokenizer=replace(config.tokenizer, vocab_size=64),
        dimensions=replace(
            config.dimensions,
            max_sequence_length=16,
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            ffn_hidden_size=32,
        ),
        expected_parameter_count=3_376,
    )
    return AtomLLM(config).eval()


def test_safetensors_roundtrip_preserves_logits_and_tied_weights(
    tmp_path: Path,
) -> None:
    torch.manual_seed(91)
    model = tiny_model()
    input_ids = torch.tensor([[2, 15, 16, 3]])
    expected = model(input_ids).logits

    checkpoint = save_safetensors_checkpoint(model, tmp_path / "model.safetensors")
    restored = tiny_model()
    load_safetensors_checkpoint(restored, checkpoint)

    assert restored.lm_head.weight is restored.token_embeddings.weight
    torch.testing.assert_close(restored(input_ids).logits, expected, rtol=0, atol=0)


def test_checkpoint_rejects_wrong_suffix_missing_file_and_model_identity(
    tmp_path: Path,
) -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match=".safetensors"):
        save_safetensors_checkpoint(model, tmp_path / "model.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_safetensors_checkpoint(model, tmp_path / "missing.safetensors")

    checkpoint = save_safetensors_checkpoint(model, tmp_path / "model.safetensors")
    incompatible = tiny_model()
    incompatible.config = replace(incompatible.config, name="different-model")
    with pytest.raises(ValueError, match="model_name"):
        load_safetensors_checkpoint(incompatible, checkpoint)
