"""Strict configuration contract for AtomLLM tokenizer training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TOKENIZER_SCHEMA_VERSION = 1
TOKENIZER_VOCAB_SIZE = 32_000
TOKENIZER_MODEL_MAX_LENGTH = 8_192
VALID_STATUSES = frozenset({"smoke", "release"})
VALID_EVALUATION_SUITES = frozenset(
    {
        "zh-Hans",
        "en",
        "zh-Hant",
        "ja",
        "code",
        "math",
        "digits",
        "whitespace",
    }
)
EXPECTED_SPECIAL_TOKENS = (
    (0, "<pad>", "padding"),
    (1, "<unk>", "unknown"),
    (2, "<bos>", "beginning_of_sequence"),
    (3, "<eos>", "end_of_sequence"),
    (4, "<|system|>", "system_role"),
    (5, "<|user|>", "user_role"),
    (6, "<|assistant|>", "assistant_role"),
    (7, "<|tool|>", "tool_role"),
    (8, "<|end_of_turn|>", "end_of_turn"),
    (9, "<|tool_call|>", "tool_call_start"),
    (10, "<|/tool_call|>", "tool_call_end"),
    (11, "<|tool_result|>", "tool_result_start"),
    (12, "<|/tool_result|>", "tool_result_end"),
    (13, "<|think|>", "thinking_start"),
    (14, "<|/think|>", "thinking_end"),
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_DATA_VERSION_PATTERN = re.compile(r"^data-[a-z0-9-]+-[0-9a-f]{12}$")


class TokenizerConfigError(ValueError):
    """Raised when a tokenizer configuration violates the frozen contract."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TokenizerConfigError(f"{context} must be a mapping with string keys")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise TokenizerConfigError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise TokenizerConfigError(
            f"{context} has unknown fields: {', '.join(unknown)}"
        )


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TokenizerConfigError(f"{field_name} must be a non-empty string")
    return value


def _safe_relative_path(value: Any, field_name: str) -> Path:
    raw_path = _non_empty_string(value, field_name)
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise TokenizerConfigError(f"{field_name} must be a safe relative path")
    return path


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    model_type: str
    vocab_size: int
    normalization: str
    pre_tokenizer: str
    decoder: str
    min_frequency: int
    dropout: float
    add_prefix_space: bool
    trim_offsets: bool
    use_regex: bool
    byte_fallback: bool
    fuse_unk: bool
    ignore_merges: bool
    max_token_length: int

    @classmethod
    def from_mapping(cls, value: Any) -> AlgorithmConfig:
        data = _mapping(value, "algorithm")
        _exact_keys(
            data,
            {
                "model_type",
                "vocab_size",
                "normalization",
                "pre_tokenizer",
                "decoder",
                "min_frequency",
                "dropout",
                "add_prefix_space",
                "trim_offsets",
                "use_regex",
                "byte_fallback",
                "fuse_unk",
                "ignore_merges",
                "max_token_length",
            },
            "algorithm",
        )
        expected_strings = {
            "model_type": "byte_level_bpe",
            "normalization": "nfc",
            "pre_tokenizer": "byte_level",
            "decoder": "byte_level",
        }
        for field_name, expected in expected_strings.items():
            if data[field_name] != expected:
                raise TokenizerConfigError(
                    f"algorithm.{field_name} must be {expected!r}"
                )
        if type(data["vocab_size"]) is not int or data["vocab_size"] != (
            TOKENIZER_VOCAB_SIZE
        ):
            raise TokenizerConfigError(
                f"algorithm.vocab_size must be {TOKENIZER_VOCAB_SIZE}"
            )
        if type(data["min_frequency"]) is not int or data["min_frequency"] < 1:
            raise TokenizerConfigError("algorithm.min_frequency must be positive")
        dropout = data["dropout"]
        if type(dropout) not in {int, float} or float(dropout) != 0.0:
            raise TokenizerConfigError(
                "algorithm.dropout must be 0.0 for deterministic training"
            )
        for field_name in ("add_prefix_space", "trim_offsets"):
            if type(data[field_name]) is not bool or data[field_name] is not False:
                raise TokenizerConfigError(f"algorithm.{field_name} must be false")
        if data["use_regex"] is not True:
            raise TokenizerConfigError("algorithm.use_regex must be true")
        for field_name in ("byte_fallback", "fuse_unk", "ignore_merges"):
            if type(data[field_name]) is not bool or data[field_name] is not False:
                raise TokenizerConfigError(f"algorithm.{field_name} must be false")
        max_token_length = data["max_token_length"]
        if type(max_token_length) is not int or max_token_length <= 0:
            raise TokenizerConfigError(
                "algorithm.max_token_length must be a positive integer"
            )
        return cls(
            model_type=data["model_type"],
            vocab_size=data["vocab_size"],
            normalization=data["normalization"],
            pre_tokenizer=data["pre_tokenizer"],
            decoder=data["decoder"],
            min_frequency=data["min_frequency"],
            dropout=float(dropout),
            add_prefix_space=data["add_prefix_space"],
            trim_offsets=data["trim_offsets"],
            use_regex=data["use_regex"],
            byte_fallback=data["byte_fallback"],
            fuse_unk=data["fuse_unk"],
            ignore_merges=data["ignore_merges"],
            max_token_length=max_token_length,
        )


@dataclass(frozen=True, slots=True)
class SpecialToken:
    token_id: int
    token: str
    purpose: str

    @classmethod
    def from_mapping(cls, value: Any) -> SpecialToken:
        data = _mapping(value, "special token")
        _exact_keys(data, {"id", "token", "purpose"}, "special token")
        if type(data["id"]) is not int or data["id"] < 0:
            raise TokenizerConfigError("special token id must be non-negative")
        return cls(
            token_id=data["id"],
            token=_non_empty_string(data["token"], "special token.token"),
            purpose=_non_empty_string(data["purpose"], "special token.purpose"),
        )


@dataclass(frozen=True, slots=True)
class TrainingDataConfig:
    data_version_id: str
    split: str
    document_count: int
    expected_sha256: str
    input_path: Path

    @classmethod
    def from_mapping(cls, value: Any) -> TrainingDataConfig:
        data = _mapping(value, "training_data")
        _exact_keys(
            data,
            {
                "data_version_id",
                "split",
                "document_count",
                "expected_sha256",
                "input_path",
            },
            "training_data",
        )
        data_version_id = _non_empty_string(
            data["data_version_id"], "training_data.data_version_id"
        )
        if _DATA_VERSION_PATTERN.fullmatch(data_version_id) is None:
            raise TokenizerConfigError(
                "training_data.data_version_id has an invalid format"
            )
        if data["split"] != "train":
            raise TokenizerConfigError("training_data.split must be 'train'")
        if type(data["document_count"]) is not int or data["document_count"] <= 0:
            raise TokenizerConfigError(
                "training_data.document_count must be a positive integer"
            )
        expected_sha256 = data["expected_sha256"]
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise TokenizerConfigError(
                "training_data.expected_sha256 must be 64 lowercase hex digits"
            )
        return cls(
            data_version_id=data_version_id,
            split=data["split"],
            document_count=data["document_count"],
            expected_sha256=expected_sha256,
            input_path=_safe_relative_path(
                data["input_path"], "training_data.input_path"
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    roundtrip_required: bool
    max_unknown_rate: float
    suites: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> EvaluationConfig:
        data = _mapping(value, "evaluation")
        _exact_keys(
            data,
            {"roundtrip_required", "max_unknown_rate", "suites"},
            "evaluation",
        )
        if data["roundtrip_required"] is not True:
            raise TokenizerConfigError("evaluation.roundtrip_required must be true")
        unknown_rate = data["max_unknown_rate"]
        if type(unknown_rate) not in {int, float} or float(unknown_rate) != 0.0:
            raise TokenizerConfigError("evaluation.max_unknown_rate must be 0.0")
        suites = data["suites"]
        if (
            not isinstance(suites, list)
            or not suites
            or not all(isinstance(item, str) for item in suites)
        ):
            raise TokenizerConfigError(
                "evaluation.suites must be a non-empty string list"
            )
        if len(suites) != len(set(suites)):
            raise TokenizerConfigError("evaluation.suites must not contain duplicates")
        invalid = sorted(set(suites) - VALID_EVALUATION_SUITES)
        if invalid:
            raise TokenizerConfigError(
                f"evaluation.suites contains unsupported values: {', '.join(invalid)}"
            )
        return cls(
            roundtrip_required=True,
            max_unknown_rate=0.0,
            suites=tuple(suites),
        )


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    schema_version: int
    name: str
    status: str
    training_eligible: bool
    model_max_length: int
    algorithm: AlgorithmConfig
    special_tokens: tuple[SpecialToken, ...]
    training_data: TrainingDataConfig
    evaluation: EvaluationConfig
    output_dir: Path

    @classmethod
    def from_mapping(cls, value: Any) -> TokenizerConfig:
        data = _mapping(value, "tokenizer config")
        _exact_keys(
            data,
            {
                "schema_version",
                "name",
                "status",
                "training_eligible",
                "model_max_length",
                "algorithm",
                "special_tokens",
                "training_data",
                "evaluation",
                "output_dir",
            },
            "tokenizer config",
        )
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != TOKENIZER_SCHEMA_VERSION
        ):
            raise TokenizerConfigError(
                f"schema_version must be {TOKENIZER_SCHEMA_VERSION}"
            )
        name = _non_empty_string(data["name"], "name")
        if _NAME_PATTERN.fullmatch(name) is None:
            raise TokenizerConfigError(
                "name must contain lowercase letters, digits, and hyphens"
            )
        status = data["status"]
        if not isinstance(status, str) or status not in VALID_STATUSES:
            raise TokenizerConfigError("status must be 'smoke' or 'release'")
        if type(data["training_eligible"]) is not bool:
            raise TokenizerConfigError("training_eligible must be a boolean")
        if status == "smoke" and data["training_eligible"] is not False:
            raise TokenizerConfigError("smoke tokenizer must not be training eligible")
        if status == "release" and data["training_eligible"] is not True:
            raise TokenizerConfigError("release tokenizer must be training eligible")
        if (
            type(data["model_max_length"]) is not int
            or data["model_max_length"] != TOKENIZER_MODEL_MAX_LENGTH
        ):
            raise TokenizerConfigError(
                f"model_max_length must be {TOKENIZER_MODEL_MAX_LENGTH}"
            )
        raw_special_tokens = data["special_tokens"]
        if not isinstance(raw_special_tokens, list):
            raise TokenizerConfigError("special_tokens must be a list")
        special_tokens = tuple(
            SpecialToken.from_mapping(item) for item in raw_special_tokens
        )
        actual_special_tokens = tuple(
            (item.token_id, item.token, item.purpose) for item in special_tokens
        )
        if actual_special_tokens != EXPECTED_SPECIAL_TOKENS:
            raise TokenizerConfigError(
                "special_tokens must exactly match the Tokenizer v1 protocol"
            )
        return cls(
            schema_version=TOKENIZER_SCHEMA_VERSION,
            name=name,
            status=status,
            training_eligible=data["training_eligible"],
            model_max_length=data["model_max_length"],
            algorithm=AlgorithmConfig.from_mapping(data["algorithm"]),
            special_tokens=special_tokens,
            training_data=TrainingDataConfig.from_mapping(data["training_data"]),
            evaluation=EvaluationConfig.from_mapping(data["evaluation"]),
            output_dir=_safe_relative_path(data["output_dir"], "output_dir"),
        )


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    """Load and strictly validate a Tokenizer v1 YAML configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"tokenizer config not found: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise TokenizerConfigError(f"invalid tokenizer YAML: {error}") from error
    return TokenizerConfig.from_mapping(value)
