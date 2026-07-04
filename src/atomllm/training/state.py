"""Validated JSON state carried by exact-resume training checkpoints."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


TRAINER_STATE_FORMAT_VERSION = 1
DATA_STATE_FORMAT_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOGICAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainingStateError(ValueError):
    """Raised when resumable training state is malformed or inconsistent."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrainingStateError(f"{context} must be a mapping with string keys")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise TrainingStateError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise TrainingStateError(f"{context} has unknown fields: {', '.join(unknown)}")


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingStateError(f"{field_name} must be a non-negative integer")
    return value


def _non_negative_float(value: Any, field_name: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise TrainingStateError(f"{field_name} must be a non-negative finite number")
    return float(value)


def _logical_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _LOGICAL_ID_PATTERN.fullmatch(value) is None:
        raise TrainingStateError(f"{field_name} must be a safe logical ID")
    return value


@dataclass(frozen=True, slots=True)
class TrainerState:
    format_version: int
    global_step: int
    micro_step: int
    samples_seen: int
    tokens_seen: int
    optimizer_steps: int
    skipped_steps: int
    current_learning_rate: float
    elapsed_training_seconds: float

    @classmethod
    def from_mapping(cls, value: Any) -> TrainerState:
        data = _mapping(value, "trainer state")
        _exact_keys(
            data,
            {
                "format_version",
                "global_step",
                "micro_step",
                "samples_seen",
                "tokens_seen",
                "optimizer_steps",
                "skipped_steps",
                "current_learning_rate",
                "elapsed_training_seconds",
            },
            "trainer state",
        )
        if (
            type(data["format_version"]) is not int
            or data["format_version"] != TRAINER_STATE_FORMAT_VERSION
        ):
            raise TrainingStateError(
                f"trainer format_version must be {TRAINER_STATE_FORMAT_VERSION}"
            )
        global_step = _non_negative_int(data["global_step"], "global_step")
        micro_step = _non_negative_int(data["micro_step"], "micro_step")
        if micro_step != 0:
            raise TrainingStateError("micro_step must be 0 at a checkpoint boundary")
        optimizer_steps = _non_negative_int(
            data["optimizer_steps"],
            "optimizer_steps",
        )
        if optimizer_steps != global_step:
            raise TrainingStateError("optimizer_steps must equal global_step")
        return cls(
            format_version=TRAINER_STATE_FORMAT_VERSION,
            global_step=global_step,
            micro_step=0,
            samples_seen=_non_negative_int(data["samples_seen"], "samples_seen"),
            tokens_seen=_non_negative_int(data["tokens_seen"], "tokens_seen"),
            optimizer_steps=optimizer_steps,
            skipped_steps=_non_negative_int(data["skipped_steps"], "skipped_steps"),
            current_learning_rate=_non_negative_float(
                data["current_learning_rate"],
                "current_learning_rate",
            ),
            elapsed_training_seconds=_non_negative_float(
                data["elapsed_training_seconds"],
                "elapsed_training_seconds",
            ),
        )

    def to_mapping(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SamplerState:
    seed: int
    position: int

    @classmethod
    def from_mapping(cls, value: Any) -> SamplerState:
        data = _mapping(value, "sampler_state")
        _exact_keys(data, {"seed", "position"}, "sampler_state")
        return cls(
            seed=_non_negative_int(data["seed"], "sampler_state.seed"),
            position=_non_negative_int(
                data["position"],
                "sampler_state.position",
            ),
        )


@dataclass(frozen=True, slots=True)
class DataState:
    format_version: int
    dataset_id: str
    dataset_manifest_sha256: str
    split: str
    epoch: int
    shard_index: int
    shard_id: str
    sample_index: int
    token_offset: int
    sampler_state: SamplerState

    @classmethod
    def from_mapping(cls, value: Any) -> DataState:
        data = _mapping(value, "data state")
        _exact_keys(
            data,
            {
                "format_version",
                "dataset_id",
                "dataset_manifest_sha256",
                "split",
                "epoch",
                "shard_index",
                "shard_id",
                "sample_index",
                "token_offset",
                "sampler_state",
            },
            "data state",
        )
        if (
            type(data["format_version"]) is not int
            or data["format_version"] != DATA_STATE_FORMAT_VERSION
        ):
            raise TrainingStateError(
                f"data format_version must be {DATA_STATE_FORMAT_VERSION}"
            )
        manifest_sha256 = data["dataset_manifest_sha256"]
        if (
            not isinstance(manifest_sha256, str)
            or _SHA256_PATTERN.fullmatch(manifest_sha256) is None
        ):
            raise TrainingStateError(
                "dataset_manifest_sha256 must be 64 lowercase hex digits"
            )
        if data["split"] != "train":
            raise TrainingStateError("split must be 'train'")
        return cls(
            format_version=DATA_STATE_FORMAT_VERSION,
            dataset_id=_logical_id(data["dataset_id"], "dataset_id"),
            dataset_manifest_sha256=manifest_sha256,
            split="train",
            epoch=_non_negative_int(data["epoch"], "epoch"),
            shard_index=_non_negative_int(data["shard_index"], "shard_index"),
            shard_id=_logical_id(data["shard_id"], "shard_id"),
            sample_index=_non_negative_int(data["sample_index"], "sample_index"),
            token_offset=_non_negative_int(data["token_offset"], "token_offset"),
            sampler_state=SamplerState.from_mapping(data["sampler_state"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        return result
