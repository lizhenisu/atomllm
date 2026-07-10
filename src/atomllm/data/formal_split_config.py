"""Reusable configuration contract for formal train/validation data splits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class FormalSplitConfigError(ValueError):
    """Raised when a formal data split configuration is unsafe or ambiguous."""


@dataclass(frozen=True, slots=True)
class FormalSplitConfig:
    train_fraction: float
    validation_fraction: float
    min_estimated_tokens: int
    max_estimated_tokens: int
    validation_quality_warnings: str
    stable_tie_breaker: str


def load_formal_split_config(path: str | Path) -> FormalSplitConfig:
    """Load a version-specific formal split recipe without hard-coding its scale."""
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FormalSplitConfigError("processing config must be a mapping")
    splits = raw.get("splits")
    selection = raw.get("validation_selection")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise FormalSplitConfigError("splits must contain only train and validation")
    train, validation = splits["train"], splits["validation"]
    if not all(type(value) is float and 0 < value < 1 for value in (train, validation)):
        raise FormalSplitConfigError(
            "split fractions must be floats between zero and one"
        )
    if not math.isclose(train + validation, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise FormalSplitConfigError("split fractions must sum to one")
    if not isinstance(selection, dict):
        raise FormalSplitConfigError("validation_selection must be a mapping")
    minimum, maximum = (
        selection.get("min_estimated_tokens"),
        selection.get("max_estimated_tokens"),
    )
    if type(minimum) is not int or type(maximum) is not int or not 0 < minimum <= maximum:
        raise FormalSplitConfigError("validation token range must be positive and ordered")
    quality = selection.get("quality_warnings")
    tie_breaker = selection.get("stable_tie_breaker")
    if not isinstance(quality, str) or not quality:
        raise FormalSplitConfigError("validation quality_warnings must be a non-empty string")
    if tie_breaker != "sha256":
        raise FormalSplitConfigError("validation stable_tie_breaker must be 'sha256'")
    return FormalSplitConfig(
        train,
        validation,
        minimum,
        maximum,
        quality,
        tie_breaker,
    )
