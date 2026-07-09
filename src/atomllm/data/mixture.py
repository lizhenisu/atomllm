"""Validated pretraining token budgets and mixture targets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from atomllm.data.schema import SCHEMA_VERSION, VALID_CONTENT_TYPES


_PLAN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_BUCKET_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_LANGUAGE_PRIORITY = ("zh-Hans", "en", "zh-Hant", "ja", "other")
_LANGUAGE_BUCKETS = frozenset(_LANGUAGE_PRIORITY)
_PRETRAINING_CONTENT_TYPES = VALID_CONTENT_TYPES - {"conversation"}
_QUALITY_BUCKETS = frozenset({"high", "standard", "exploratory"})
_BUDGET_NAMES = ("smoke", "pilot", "main", "stretch")


class MixtureConfigError(ValueError):
    """Raised when a pretraining mixture configuration is invalid."""


def _require_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise MixtureConfigError(f"{context} must be a mapping with string keys")
    return data


def _require_exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise MixtureConfigError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise MixtureConfigError(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_fraction(value: Any, field_name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixtureConfigError(f"{field_name} must be a number")
    fraction = float(value)
    lower_bound_valid = fraction >= 0 if allow_zero else fraction > 0
    if not math.isfinite(fraction) or not lower_bound_valid or fraction > 1:
        qualifier = "between 0 and 1" if allow_zero else "greater than 0 and at most 1"
        raise MixtureConfigError(f"{field_name} must be {qualifier}")
    return fraction


def _parse_distribution(
    data: Any, expected: frozenset[str], context: str
) -> dict[str, float]:
    mapping = _require_mapping(data, context)
    _require_exact_keys(mapping, set(expected), context)
    weights = {
        key: _require_fraction(value, f"{context}.{key}", allow_zero=False)
        for key, value in mapping.items()
    }
    if not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-9):
        raise MixtureConfigError(f"{context} fractions must sum to 1")
    return weights


@dataclass(frozen=True, slots=True)
class TokenBudgets:
    smoke: int
    pilot: int
    main: int
    stretch: int

    @classmethod
    def from_mapping(cls, data: Any) -> TokenBudgets:
        mapping = _require_mapping(data, "budgets")
        _require_exact_keys(mapping, set(_BUDGET_NAMES), "budgets")
        values: list[int] = []
        for name in _BUDGET_NAMES:
            value = mapping[name]
            if type(value) is not int or value <= 0:
                raise MixtureConfigError(f"budgets.{name} must be a positive integer")
            values.append(value)
        if values != sorted(values) or len(values) != len(set(values)):
            raise MixtureConfigError(
                "budgets must increase strictly: smoke < pilot < main < stretch"
            )
        return cls(**dict(zip(_BUDGET_NAMES, values, strict=True)))

    def to_mapping(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _BUDGET_NAMES}


@dataclass(frozen=True, slots=True)
class LengthBucket:
    name: str
    min_tokens: int
    max_tokens: int
    fraction: float

    @classmethod
    def from_mapping(cls, data: Any) -> LengthBucket:
        mapping = _require_mapping(data, "length bucket")
        _require_exact_keys(
            mapping,
            {"name", "min_tokens", "max_tokens", "fraction"},
            "length bucket",
        )
        name = mapping["name"]
        if not isinstance(name, str) or _BUCKET_NAME_PATTERN.fullmatch(name) is None:
            raise MixtureConfigError("length bucket name is invalid")
        min_tokens = mapping["min_tokens"]
        max_tokens = mapping["max_tokens"]
        if (
            type(min_tokens) is not int
            or type(max_tokens) is not int
            or min_tokens <= 0
            or max_tokens < min_tokens
        ):
            raise MixtureConfigError(
                "length bucket token bounds must be positive ordered integers"
            )
        fraction = _require_fraction(
            mapping["fraction"], f"length_mix.{name}.fraction", allow_zero=False
        )
        return cls(
            name=name,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            fraction=fraction,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "fraction": self.fraction,
        }


def _parse_length_mix(data: Any) -> tuple[LengthBucket, ...]:
    if not isinstance(data, list) or not data:
        raise MixtureConfigError("length_mix must be a non-empty list")
    buckets = tuple(LengthBucket.from_mapping(item) for item in data)
    names = [bucket.name for bucket in buckets]
    if len(names) != len(set(names)):
        raise MixtureConfigError("length bucket names must be unique")
    if buckets[0].min_tokens != 1:
        raise MixtureConfigError("length_mix must start at 1 token")
    for previous, current in zip(buckets, buckets[1:], strict=False):
        if current.min_tokens != previous.max_tokens + 1:
            raise MixtureConfigError("length_mix token ranges must be contiguous")
    if buckets[-1].max_tokens != 8192:
        raise MixtureConfigError("length_mix must end at 8192 tokens")
    if not math.isclose(
        math.fsum(bucket.fraction for bucket in buckets), 1.0, abs_tol=1e-9
    ):
        raise MixtureConfigError("length_mix fractions must sum to 1")
    return buckets


@dataclass(frozen=True, slots=True)
class MixtureConstraints:
    max_source_fraction: float
    max_exact_duplicate_fraction: float
    max_near_duplicate_fraction: float
    privacy_action: str
    unknown_license_action: str

    @classmethod
    def from_mapping(cls, data: Any) -> MixtureConstraints:
        mapping = _require_mapping(data, "constraints")
        _require_exact_keys(
            mapping,
            {
                "max_source_fraction",
                "max_exact_duplicate_fraction",
                "max_near_duplicate_fraction",
                "privacy_action",
                "unknown_license_action",
            },
            "constraints",
        )
        max_source_fraction = _require_fraction(
            mapping["max_source_fraction"],
            "constraints.max_source_fraction",
            allow_zero=False,
        )
        max_exact_duplicate_fraction = _require_fraction(
            mapping["max_exact_duplicate_fraction"],
            "constraints.max_exact_duplicate_fraction",
            allow_zero=True,
        )
        max_near_duplicate_fraction = _require_fraction(
            mapping["max_near_duplicate_fraction"],
            "constraints.max_near_duplicate_fraction",
            allow_zero=True,
        )
        if max_exact_duplicate_fraction > max_near_duplicate_fraction:
            raise MixtureConfigError(
                "exact duplicate limit cannot exceed near duplicate limit"
            )
        if mapping["privacy_action"] != "warn":
            raise MixtureConfigError("constraints.privacy_action must be 'warn'")
        if mapping["unknown_license_action"] != "warn":
            raise MixtureConfigError(
                "constraints.unknown_license_action must be 'warn'"
            )
        return cls(
            max_source_fraction=max_source_fraction,
            max_exact_duplicate_fraction=max_exact_duplicate_fraction,
            max_near_duplicate_fraction=max_near_duplicate_fraction,
            privacy_action="warn",
            unknown_license_action="warn",
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "max_source_fraction": self.max_source_fraction,
            "max_exact_duplicate_fraction": self.max_exact_duplicate_fraction,
            "max_near_duplicate_fraction": self.max_near_duplicate_fraction,
            "privacy_action": self.privacy_action,
            "unknown_license_action": self.unknown_license_action,
        }


@dataclass(frozen=True, slots=True)
class PretrainingMixture:
    schema_version: int
    plan_id: str
    model_id: str
    budgets: TokenBudgets
    language_mix: dict[str, float]
    content_mix: dict[str, float]
    quality_mix: dict[str, float]
    length_mix: tuple[LengthBucket, ...]
    constraints: MixtureConstraints

    @classmethod
    def from_mapping(cls, data: Any) -> PretrainingMixture:
        mapping = _require_mapping(data, "pretraining mixture")
        _require_exact_keys(
            mapping,
            {
                "schema_version",
                "plan_id",
                "model_id",
                "budgets",
                "language_mix",
                "content_mix",
                "quality_mix",
                "length_mix",
                "constraints",
            },
            "pretraining mixture",
        )
        if type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1:
            raise MixtureConfigError("schema_version must be 1")
        plan_id = mapping["plan_id"]
        if not isinstance(plan_id, str) or _PLAN_ID_PATTERN.fullmatch(plan_id) is None:
            raise MixtureConfigError("plan_id must be a lowercase path-safe identifier")
        model_id = mapping["model_id"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise MixtureConfigError("model_id must be a non-empty string")

        language_mix = _parse_distribution(
            mapping["language_mix"], _LANGUAGE_BUCKETS, "language_mix"
        )
        language_weights = [language_mix[name] for name in _LANGUAGE_PRIORITY]
        if any(
            current <= following
            for current, following in zip(
                language_weights, language_weights[1:], strict=False
            )
        ):
            raise MixtureConfigError(
                "language_mix must satisfy zh-Hans > en > zh-Hant > ja > other"
            )

        return cls(
            schema_version=SCHEMA_VERSION,
            plan_id=plan_id,
            model_id=model_id,
            budgets=TokenBudgets.from_mapping(mapping["budgets"]),
            language_mix=language_mix,
            content_mix=_parse_distribution(
                mapping["content_mix"], _PRETRAINING_CONTENT_TYPES, "content_mix"
            ),
            quality_mix=_parse_distribution(
                mapping["quality_mix"], _QUALITY_BUCKETS, "quality_mix"
            ),
            length_mix=_parse_length_mix(mapping["length_mix"]),
            constraints=MixtureConstraints.from_mapping(mapping["constraints"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "model_id": self.model_id,
            "budgets": self.budgets.to_mapping(),
            "language_mix": dict(self.language_mix),
            "content_mix": dict(self.content_mix),
            "quality_mix": dict(self.quality_mix),
            "length_mix": [bucket.to_mapping() for bucket in self.length_mix],
            "constraints": self.constraints.to_mapping(),
        }


def load_pretraining_mixture(path: str | Path) -> PretrainingMixture:
    mixture_path = Path(path)
    if not mixture_path.is_file():
        raise FileNotFoundError(f"pretraining mixture not found: {mixture_path}")
    try:
        raw_data = yaml.safe_load(mixture_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise MixtureConfigError(
            f"invalid pretraining mixture YAML: {error}"
        ) from error
    return PretrainingMixture.from_mapping(raw_data)
