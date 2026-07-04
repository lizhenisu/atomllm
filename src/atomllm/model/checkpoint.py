"""Safetensors checkpoint I/O for AtomLLM models with tied weights."""

from __future__ import annotations

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_model, save_model

from atomllm.model.model import AtomLLM


CHECKPOINT_FORMAT = "atomllm-safetensors-v1"


def save_safetensors_checkpoint(model: AtomLLM, path: str | Path) -> Path:
    """Save model weights and architecture identity without duplicating tied weights."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("checkpoint path must end with .safetensors")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "model_name": model.config.name,
        "parameter_count": str(model.config.expected_parameter_count),
    }
    save_model(model, str(checkpoint_path), metadata=metadata)
    return checkpoint_path


def load_safetensors_checkpoint(model: AtomLLM, path: str | Path) -> None:
    """Load weights after validating checkpoint format and architecture identity."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    with safe_open(checkpoint_path, framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    expected = {
        "format": CHECKPOINT_FORMAT,
        "model_name": model.config.name,
        "parameter_count": str(model.config.expected_parameter_count),
    }
    mismatches = {
        key: (metadata.get(key), expected_value)
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in sorted(mismatches.items())
        )
        raise ValueError(f"checkpoint metadata mismatch: {details}")
    device = next(model.parameters()).device
    load_model(model, checkpoint_path, strict=True, device=str(device))
