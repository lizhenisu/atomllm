"""Strict AtomLLM model configuration and exact parameter accounting."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MODEL_SCHEMA_VERSION = 1
CORE_SPECIAL_TOKEN_IDS = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_VERSION_PATTERN = re.compile(r"^tokenizer-version-[a-z0-9-]+-[0-9a-f]{12}$")


class ModelConfigError(ValueError):
    """Raised when a model configuration violates the architecture contract."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ModelConfigError(f"{context} must be a mapping with string keys")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ModelConfigError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise ModelConfigError(f"{context} has unknown fields: {', '.join(unknown)}")


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ModelConfigError(f"{field_name} must be a positive integer")
    return value


def _finite_float(value: Any, field_name: str, *, positive: bool) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ModelConfigError(f"{field_name} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ModelConfigError(f"{field_name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class TokenizerBinding:
    version_id: str
    tokenizer_sha256: str
    vocab_size: int
    special_token_ids: dict[str, int]

    @classmethod
    def from_mapping(cls, value: Any) -> TokenizerBinding:
        data = _mapping(value, "tokenizer")
        _exact_keys(
            data,
            {
                "version_id",
                "tokenizer_sha256",
                "vocab_size",
                "special_token_ids",
            },
            "tokenizer",
        )
        version_id = data["version_id"]
        if (
            not isinstance(version_id, str)
            or _TOKENIZER_VERSION_PATTERN.fullmatch(version_id) is None
        ):
            raise ModelConfigError("tokenizer.version_id has an invalid format")
        tokenizer_sha256 = data["tokenizer_sha256"]
        if (
            not isinstance(tokenizer_sha256, str)
            or _SHA256_PATTERN.fullmatch(tokenizer_sha256) is None
        ):
            raise ModelConfigError(
                "tokenizer.tokenizer_sha256 must be 64 lowercase hex digits"
            )
        vocab_size = _positive_int(data["vocab_size"], "tokenizer.vocab_size")
        special_token_ids = _mapping(
            data["special_token_ids"], "tokenizer.special_token_ids"
        )
        _exact_keys(
            special_token_ids,
            set(CORE_SPECIAL_TOKEN_IDS),
            "tokenizer.special_token_ids",
        )
        if any(type(token_id) is not int for token_id in special_token_ids.values()):
            raise ModelConfigError(
                "tokenizer.special_token_ids values must be integers"
            )
        if special_token_ids != CORE_SPECIAL_TOKEN_IDS:
            raise ModelConfigError(
                "tokenizer.special_token_ids must match the Tokenizer v1 protocol"
            )
        return cls(
            version_id=version_id,
            tokenizer_sha256=tokenizer_sha256,
            vocab_size=vocab_size,
            special_token_ids=dict(special_token_ids),
        )


@dataclass(frozen=True, slots=True)
class ModelDimensions:
    max_sequence_length: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    ffn_hidden_size: int

    @classmethod
    def from_mapping(cls, value: Any) -> ModelDimensions:
        data = _mapping(value, "dimensions")
        expected = {
            "max_sequence_length",
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "ffn_hidden_size",
        }
        _exact_keys(data, expected, "dimensions")
        values = {
            field_name: _positive_int(data[field_name], f"dimensions.{field_name}")
            for field_name in expected
        }
        if values["hidden_size"] != values["num_attention_heads"] * values["head_dim"]:
            raise ModelConfigError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if values["num_attention_heads"] % values["num_key_value_heads"] != 0:
            raise ModelConfigError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ModelComponents:
    normalization: str
    rms_norm_epsilon: float
    activation: str
    position_encoding: str
    rope_theta: float
    rope_scaling: None
    tie_word_embeddings: bool
    use_bias: bool
    attention_dropout: float
    residual_dropout: float
    initializer_std: float
    scale_residual_projections: bool
    default_dtype: str

    @classmethod
    def from_mapping(cls, value: Any) -> ModelComponents:
        data = _mapping(value, "components")
        _exact_keys(
            data,
            {
                "normalization",
                "rms_norm_epsilon",
                "activation",
                "position_encoding",
                "rope_theta",
                "rope_scaling",
                "tie_word_embeddings",
                "use_bias",
                "attention_dropout",
                "residual_dropout",
                "initializer_std",
                "scale_residual_projections",
                "default_dtype",
            },
            "components",
        )
        expected_strings = {
            "normalization": "rmsnorm",
            "activation": "swiglu",
            "position_encoding": "rope",
            "default_dtype": "bfloat16",
        }
        for field_name, expected in expected_strings.items():
            if data[field_name] != expected:
                raise ModelConfigError(f"components.{field_name} must be {expected!r}")
        if data["rope_scaling"] is not None:
            raise ModelConfigError("components.rope_scaling must be null for v1")
        for field_name, expected in (
            ("tie_word_embeddings", True),
            ("use_bias", False),
            ("scale_residual_projections", True),
        ):
            if type(data[field_name]) is not bool or data[field_name] is not expected:
                raise ModelConfigError(
                    f"components.{field_name} must be {str(expected).lower()}"
                )
        attention_dropout = _finite_float(
            data["attention_dropout"],
            "components.attention_dropout",
            positive=False,
        )
        residual_dropout = _finite_float(
            data["residual_dropout"],
            "components.residual_dropout",
            positive=False,
        )
        for field_name, dropout in (
            ("attention_dropout", attention_dropout),
            ("residual_dropout", residual_dropout),
        ):
            if not 0.0 <= dropout < 1.0:
                raise ModelConfigError(f"components.{field_name} must be in [0, 1)")
            if dropout != 0.0:
                raise ModelConfigError(
                    f"components.{field_name} must be 0.0 for model v1"
                )
        return cls(
            normalization=data["normalization"],
            rms_norm_epsilon=_finite_float(
                data["rms_norm_epsilon"],
                "components.rms_norm_epsilon",
                positive=True,
            ),
            activation=data["activation"],
            position_encoding=data["position_encoding"],
            rope_theta=_finite_float(
                data["rope_theta"],
                "components.rope_theta",
                positive=True,
            ),
            rope_scaling=None,
            tie_word_embeddings=data["tie_word_embeddings"],
            use_bias=data["use_bias"],
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            initializer_std=_finite_float(
                data["initializer_std"],
                "components.initializer_std",
                positive=True,
            ),
            scale_residual_projections=data["scale_residual_projections"],
            default_dtype=data["default_dtype"],
        )


@dataclass(frozen=True, slots=True)
class ParameterBreakdown:
    token_embeddings: int
    attention_per_layer: int
    attention_all_layers: int
    ffn_per_layer: int
    ffn_all_layers: int
    rmsnorm_per_layer: int
    rmsnorm_all_layers: int
    final_rmsnorm: int
    lm_head: int
    total: int

    def to_mapping(self) -> dict[str, int]:
        return {
            "token_embeddings": self.token_embeddings,
            "attention_per_layer": self.attention_per_layer,
            "attention_all_layers": self.attention_all_layers,
            "ffn_per_layer": self.ffn_per_layer,
            "ffn_all_layers": self.ffn_all_layers,
            "rmsnorm_per_layer": self.rmsnorm_per_layer,
            "rmsnorm_all_layers": self.rmsnorm_all_layers,
            "final_rmsnorm": self.final_rmsnorm,
            "lm_head": self.lm_head,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class ModelConfig:
    schema_version: int
    name: str
    status: str
    architecture: str
    tokenizer: TokenizerBinding
    dimensions: ModelDimensions
    components: ModelComponents
    expected_parameter_count: int

    @classmethod
    def from_mapping(cls, value: Any) -> ModelConfig:
        data = _mapping(value, "model config")
        _exact_keys(
            data,
            {
                "schema_version",
                "name",
                "status",
                "architecture",
                "tokenizer",
                "dimensions",
                "components",
                "expected_parameter_count",
            },
            "model config",
        )
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != MODEL_SCHEMA_VERSION
        ):
            raise ModelConfigError(f"schema_version must be {MODEL_SCHEMA_VERSION}")
        name = data["name"]
        if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
            raise ModelConfigError(
                "name must contain lowercase letters, digits, and hyphens"
            )
        if not isinstance(data["status"], str) or data["status"] not in {
            "smoke",
            "release",
        }:
            raise ModelConfigError("status must be 'smoke' or 'release'")
        if data["architecture"] != "decoder_only_transformer":
            raise ModelConfigError("architecture must be 'decoder_only_transformer'")
        expected_parameter_count = _positive_int(
            data["expected_parameter_count"],
            "expected_parameter_count",
        )
        config = cls(
            schema_version=MODEL_SCHEMA_VERSION,
            name=name,
            status=data["status"],
            architecture=data["architecture"],
            tokenizer=TokenizerBinding.from_mapping(data["tokenizer"]),
            dimensions=ModelDimensions.from_mapping(data["dimensions"]),
            components=ModelComponents.from_mapping(data["components"]),
            expected_parameter_count=expected_parameter_count,
        )
        actual_parameter_count = calculate_parameter_count(config).total
        if actual_parameter_count != expected_parameter_count:
            raise ModelConfigError(
                "expected_parameter_count does not match the architecture: "
                f"expected {expected_parameter_count}, calculated "
                f"{actual_parameter_count}"
            )
        return config


def calculate_parameter_count(config: ModelConfig) -> ParameterBreakdown:
    """Calculate exact trainable parameters for the frozen bias-free architecture."""
    dimensions = config.dimensions
    vocab_size = config.tokenizer.vocab_size
    hidden_size = dimensions.hidden_size
    query_width = dimensions.num_attention_heads * dimensions.head_dim
    key_value_width = dimensions.num_key_value_heads * dimensions.head_dim

    token_embeddings = vocab_size * hidden_size
    attention_per_layer = (
        hidden_size * query_width
        + hidden_size * key_value_width
        + hidden_size * key_value_width
        + query_width * hidden_size
    )
    ffn_per_layer = 3 * hidden_size * dimensions.ffn_hidden_size
    rmsnorm_per_layer = 2 * hidden_size
    attention_all_layers = attention_per_layer * dimensions.num_layers
    ffn_all_layers = ffn_per_layer * dimensions.num_layers
    rmsnorm_all_layers = rmsnorm_per_layer * dimensions.num_layers
    final_rmsnorm = hidden_size
    lm_head = 0 if config.components.tie_word_embeddings else (vocab_size * hidden_size)
    total = (
        token_embeddings
        + attention_all_layers
        + ffn_all_layers
        + rmsnorm_all_layers
        + final_rmsnorm
        + lm_head
    )
    return ParameterBreakdown(
        token_embeddings=token_embeddings,
        attention_per_layer=attention_per_layer,
        attention_all_layers=attention_all_layers,
        ffn_per_layer=ffn_per_layer,
        ffn_all_layers=ffn_all_layers,
        rmsnorm_per_layer=rmsnorm_per_layer,
        rmsnorm_all_layers=rmsnorm_all_layers,
        final_rmsnorm=final_rmsnorm,
        lm_head=lm_head,
        total=total,
    )


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ModelConfigError(f"invalid model YAML: {error}") from error
    return ModelConfig.from_mapping(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an AtomLLM model config and count exact parameters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/atom-base-300m.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_model_config(args.config)
    breakdown = calculate_parameter_count(config)
    print(json.dumps(breakdown.to_mapping(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
