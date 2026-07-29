"""Strict stage-4 training and experiment-matrix configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from atomllm.model.config import load_model_config


TRAINING_SCHEMA_VERSION = 1
MATRIX_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class TrainingConfigError(ValueError):
    """Raised when a stage-4 training configuration violates its contract."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrainingConfigError(f"{context} must be a mapping with string keys")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise TrainingConfigError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise TrainingConfigError(f"{context} has unknown fields: {', '.join(unknown)}")


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingConfigError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingConfigError(f"{field_name} must be a non-negative integer")
    return value


def _finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise TrainingConfigError(f"{field_name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise TrainingConfigError(f"{field_name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise TrainingConfigError(f"{field_name} must be at most {maximum}")
    return result


def _name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _NAME_PATTERN.fullmatch(value) is None:
        raise TrainingConfigError(
            f"{field_name} must contain lowercase letters, digits, and hyphens"
        )
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise TrainingConfigError(f"{field_name} must be 64 lowercase hex digits")
    return value


def _relative_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainingConfigError(f"{field_name} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingConfigError(f"{field_name} must be a safe relative path")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelBinding:
    config_path: Path
    config_sha256: str
    name: str
    expected_parameter_count: int

    @classmethod
    def from_mapping(cls, value: Any) -> ModelBinding:
        data = _mapping(value, "model")
        _exact_keys(
            data,
            {"config_path", "config_sha256", "name", "expected_parameter_count"},
            "model",
        )
        return cls(
            config_path=_relative_path(data["config_path"], "model.config_path"),
            config_sha256=_sha256(
                data["config_sha256"],
                "model.config_sha256",
            ),
            name=_name(data["name"], "model.name"),
            expected_parameter_count=_positive_int(
                data["expected_parameter_count"],
                "model.expected_parameter_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class DataBinding:
    data_version_id: str
    data_manifest_sha256: str
    split: str
    split_sha256: str
    tokenizer_version_id: str
    tokenizer_sha256: str
    formal_training_eligible: bool
    dataset_manifest_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> DataBinding:
        data = _mapping(value, "data")
        required = {
            "data_version_id",
            "data_manifest_sha256",
            "split",
            "split_sha256",
            "tokenizer_version_id",
            "tokenizer_sha256",
            "formal_training_eligible",
        }
        optional = {"dataset_manifest_sha256"}
        unknown = set(data) - required - optional
        missing = required - set(data)
        if missing or unknown:
            _exact_keys(data, required | (optional & set(data)), "data")
        for field_name in ("data_version_id", "tokenizer_version_id"):
            if not isinstance(data[field_name], str) or not data[field_name]:
                raise TrainingConfigError(f"data.{field_name} must be non-empty")
        if data["split"] != "train":
            raise TrainingConfigError("data.split must be 'train'")
        if type(data["formal_training_eligible"]) is not bool:
            raise TrainingConfigError("data.formal_training_eligible must be a boolean")
        return cls(
            data_version_id=data["data_version_id"],
            data_manifest_sha256=_sha256(
                data["data_manifest_sha256"],
                "data.data_manifest_sha256",
            ),
            split="train",
            split_sha256=_sha256(data["split_sha256"], "data.split_sha256"),
            tokenizer_version_id=data["tokenizer_version_id"],
            tokenizer_sha256=_sha256(
                data["tokenizer_sha256"],
                "data.tokenizer_sha256",
            ),
            formal_training_eligible=data["formal_training_eligible"],
            dataset_manifest_sha256=(
                None
                if data.get("dataset_manifest_sha256") is None
                else _sha256(
                    data["dataset_manifest_sha256"],
                    "data.dataset_manifest_sha256",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class BatchConfig:
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_accumulation_schedule: tuple[int, ...] = ()

    @property
    def tokens_per_micro_batch(self) -> int:
        return self.sequence_length * self.micro_batch_size

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.tokens_per_micro_batch * self.gradient_accumulation_steps

    def accumulation_steps_for_step(self, global_step: int) -> int:
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if not self.gradient_accumulation_schedule:
            return self.gradient_accumulation_steps
        return self.gradient_accumulation_schedule[
            global_step % len(self.gradient_accumulation_schedule)
        ]

    def micro_steps_through(self, optimizer_steps: int) -> int:
        if type(optimizer_steps) is not int or optimizer_steps < 0:
            raise ValueError("optimizer_steps must be a non-negative integer")
        if not self.gradient_accumulation_schedule:
            return optimizer_steps * self.gradient_accumulation_steps
        schedule = self.gradient_accumulation_schedule
        cycles, remainder = divmod(optimizer_steps, len(schedule))
        return cycles * sum(schedule) + sum(schedule[:remainder])

    def samples_through(self, optimizer_steps: int, world_size: int) -> int:
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        return (
            self.micro_steps_through(optimizer_steps)
            * self.micro_batch_size
            * world_size
        )

    def tokens_through(self, optimizer_steps: int, world_size: int) -> int:
        return self.samples_through(optimizer_steps, world_size) * self.sequence_length

    @classmethod
    def from_mapping(cls, value: Any) -> BatchConfig:
        data = _mapping(value, "batch")
        required = {
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
        }
        optional = {"gradient_accumulation_schedule"}
        unknown = set(data) - required - optional
        missing = required - set(data)
        if missing or unknown:
            _exact_keys(data, required | (optional & set(data)), "batch")
        raw_schedule = data.get("gradient_accumulation_schedule", [])
        if not isinstance(raw_schedule, list):
            raise TrainingConfigError(
                "batch.gradient_accumulation_schedule must be a list"
            )
        schedule = tuple(
            _positive_int(value, f"batch.gradient_accumulation_schedule[{index}]")
            for index, value in enumerate(raw_schedule)
        )
        configured_steps = _positive_int(
            data["gradient_accumulation_steps"],
            "batch.gradient_accumulation_steps",
        )
        if schedule and max(schedule) != configured_steps:
            raise TrainingConfigError(
                "batch.gradient_accumulation_steps must equal the schedule maximum"
            )
        return cls(
            sequence_length=_positive_int(
                data["sequence_length"],
                "batch.sequence_length",
            ),
            micro_batch_size=_positive_int(
                data["micro_batch_size"],
                "batch.micro_batch_size",
            ),
            gradient_accumulation_steps=configured_steps,
            gradient_accumulation_schedule=schedule,
        )


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float

    @classmethod
    def from_mapping(cls, value: Any) -> OptimizerConfig:
        data = _mapping(value, "optimizer")
        _exact_keys(
            data,
            {
                "name",
                "learning_rate",
                "beta1",
                "beta2",
                "epsilon",
                "weight_decay",
            },
            "optimizer",
        )
        if not isinstance(data["name"], str) or data["name"] != "adamw":
            raise TrainingConfigError("optimizer.name must be 'adamw'")
        beta1 = _finite_float(data["beta1"], "optimizer.beta1", minimum=0)
        beta2 = _finite_float(data["beta2"], "optimizer.beta2", minimum=0)
        if beta1 >= 1 or beta2 >= 1 or beta1 >= beta2:
            raise TrainingConfigError(
                "optimizer betas must satisfy 0 <= beta1 < beta2 < 1"
            )
        return cls(
            name="adamw",
            learning_rate=_finite_float(
                data["learning_rate"],
                "optimizer.learning_rate",
                minimum=1e-12,
            ),
            beta1=beta1,
            beta2=beta2,
            epsilon=_finite_float(
                data["epsilon"],
                "optimizer.epsilon",
                minimum=1e-16,
            ),
            weight_decay=_finite_float(
                data["weight_decay"],
                "optimizer.weight_decay",
                minimum=0,
            ),
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str
    warmup_steps: int
    total_steps: int
    minimum_learning_rate_ratio: float
    cooldown_steps: int = 0

    @classmethod
    def from_mapping(cls, value: Any) -> SchedulerConfig:
        data = _mapping(value, "scheduler")
        required = {
            "name",
            "warmup_steps",
            "total_steps",
            "minimum_learning_rate_ratio",
        }
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required - {"cooldown_steps"})
        if missing:
            raise TrainingConfigError(
                f"scheduler missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise TrainingConfigError(
                f"scheduler has unknown fields: {', '.join(unknown)}"
            )
        if not isinstance(data["name"], str) or data["name"] not in {
            "cosine",
            "constant",
            "trapezoidal",
        }:
            raise TrainingConfigError(
                "scheduler.name must be 'cosine', 'constant', or 'trapezoidal'"
            )
        warmup_steps = _non_negative_int(
            data["warmup_steps"],
            "scheduler.warmup_steps",
        )
        total_steps = _positive_int(data["total_steps"], "scheduler.total_steps")
        if warmup_steps >= total_steps:
            raise TrainingConfigError(
                "scheduler.warmup_steps must be less than total_steps"
            )
        cooldown_steps = _non_negative_int(
            data.get("cooldown_steps", 0),
            "scheduler.cooldown_steps",
        )
        if data["name"] == "trapezoidal":
            if cooldown_steps == 0:
                raise TrainingConfigError(
                    "trapezoidal scheduler requires cooldown_steps greater than zero"
                )
            if warmup_steps + cooldown_steps > total_steps:
                raise TrainingConfigError(
                    "scheduler warmup and cooldown exceed total_steps"
                )
        elif cooldown_steps != 0:
            raise TrainingConfigError(
                "scheduler.cooldown_steps is only valid for trapezoidal schedules"
            )
        return cls(
            name=data["name"],
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            minimum_learning_rate_ratio=_finite_float(
                data["minimum_learning_rate_ratio"],
                "scheduler.minimum_learning_rate_ratio",
                minimum=0,
                maximum=1,
            ),
            cooldown_steps=cooldown_steps,
        )


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    max_gradient_norm: float
    reject_non_finite_loss: bool
    reject_non_finite_gradient_norm: bool

    @classmethod
    def from_mapping(cls, value: Any) -> StabilityConfig:
        data = _mapping(value, "stability")
        _exact_keys(
            data,
            {
                "max_gradient_norm",
                "reject_non_finite_loss",
                "reject_non_finite_gradient_norm",
            },
            "stability",
        )
        for field_name in (
            "reject_non_finite_loss",
            "reject_non_finite_gradient_norm",
        ):
            if data[field_name] is not True:
                raise TrainingConfigError(f"stability.{field_name} must be true")
        return cls(
            max_gradient_norm=_finite_float(
                data["max_gradient_norm"],
                "stability.max_gradient_norm",
                minimum=1e-12,
            ),
            reject_non_finite_loss=True,
            reject_non_finite_gradient_norm=True,
        )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    save_every_steps: int
    keep_last: int
    exact_resume: bool
    model_format: str
    save_optimizer: bool
    save_scheduler: bool
    save_rng_state: bool
    save_data_state: bool

    @classmethod
    def from_mapping(cls, value: Any) -> CheckpointConfig:
        data = _mapping(value, "checkpoint")
        _exact_keys(
            data,
            {
                "save_every_steps",
                "keep_last",
                "exact_resume",
                "model_format",
                "save_optimizer",
                "save_scheduler",
                "save_rng_state",
                "save_data_state",
            },
            "checkpoint",
        )
        if (
            not isinstance(data["model_format"], str)
            or data["model_format"] != "safetensors"
        ):
            raise TrainingConfigError("checkpoint.model_format must be 'safetensors'")
        for field_name in (
            "exact_resume",
            "save_optimizer",
            "save_scheduler",
            "save_rng_state",
            "save_data_state",
        ):
            if data[field_name] is not True:
                raise TrainingConfigError(f"checkpoint.{field_name} must be true")
        return cls(
            save_every_steps=_positive_int(
                data["save_every_steps"],
                "checkpoint.save_every_steps",
            ),
            keep_last=_positive_int(data["keep_last"], "checkpoint.keep_last"),
            exact_resume=True,
            model_format="safetensors",
            save_optimizer=True,
            save_scheduler=True,
            save_rng_state=True,
            save_data_state=True,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str
    precision: str
    gradient_checkpointing: bool
    compile_model: bool
    deterministic: bool
    loss_chunk_size: int | None = None
    checkpoint_segment_layers: int = 1
    checkpoint_interval_segments: int = 1
    ddp_bucket_cap_mb: int = 25
    ddp_static_graph: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> RuntimeConfig:
        data = _mapping(value, "runtime")
        required = {
            "device",
            "precision",
            "gradient_checkpointing",
            "compile_model",
            "deterministic",
        }
        optional = {
            "loss_chunk_size",
            "checkpoint_segment_layers",
            "checkpoint_interval_segments",
            "ddp_bucket_cap_mb",
            "ddp_static_graph",
        }
        unknown = set(data) - required - optional
        missing = required - set(data)
        if missing or unknown:
            _exact_keys(
                data,
                required | (optional & set(data)),
                "runtime",
            )
        if not isinstance(data["device"], str) or data["device"] not in {
            "cpu",
            "cuda",
        }:
            raise TrainingConfigError("runtime.device must be 'cpu' or 'cuda'")
        if not isinstance(data["precision"], str) or data["precision"] not in {
            "fp32",
            "bf16",
        }:
            raise TrainingConfigError("runtime.precision must be 'fp32' or 'bf16'")
        for field_name in (
            "gradient_checkpointing",
            "compile_model",
            "deterministic",
        ):
            if type(data[field_name]) is not bool:
                raise TrainingConfigError(f"runtime.{field_name} must be a boolean")
        if "ddp_static_graph" in data and type(data["ddp_static_graph"]) is not bool:
            raise TrainingConfigError("runtime.ddp_static_graph must be a boolean")
        if data["device"] == "cpu" and data["precision"] != "fp32":
            raise TrainingConfigError("CPU training must use fp32")
        return cls(
            device=data["device"],
            precision=data["precision"],
            gradient_checkpointing=data["gradient_checkpointing"],
            compile_model=data["compile_model"],
            deterministic=data["deterministic"],
            loss_chunk_size=(
                None
                if data.get("loss_chunk_size") is None
                else _positive_int(data["loss_chunk_size"], "runtime.loss_chunk_size")
            ),
            checkpoint_segment_layers=(
                1
                if "checkpoint_segment_layers" not in data
                else _positive_int(
                    data["checkpoint_segment_layers"],
                    "runtime.checkpoint_segment_layers",
                )
            ),
            checkpoint_interval_segments=(
                1
                if "checkpoint_interval_segments" not in data
                else _positive_int(
                    data["checkpoint_interval_segments"],
                    "runtime.checkpoint_interval_segments",
                )
            ),
            ddp_bucket_cap_mb=(
                25
                if "ddp_bucket_cap_mb" not in data
                else _positive_int(
                    data["ddp_bucket_cap_mb"],
                    "runtime.ddp_bucket_cap_mb",
                )
            ),
            ddp_static_graph=data.get("ddp_static_graph", False),
        )


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    enabled: bool = True
    tensorboard: bool = True
    log_every_steps: int = 1
    flush_every_steps: int = 1

    @classmethod
    def from_mapping(cls, value: Any) -> MonitoringConfig:
        if value is None:
            return cls()
        data = _mapping(value, "monitoring")
        _exact_keys(
            data,
            {"enabled", "tensorboard", "log_every_steps", "flush_every_steps"},
            "monitoring",
        )
        for field_name in ("enabled", "tensorboard"):
            if type(data[field_name]) is not bool:
                raise TrainingConfigError(f"monitoring.{field_name} must be a boolean")
        return cls(
            enabled=data["enabled"],
            tensorboard=data["tensorboard"],
            log_every_steps=_positive_int(
                data["log_every_steps"], "monitoring.log_every_steps"
            ),
            flush_every_steps=_positive_int(
                data["flush_every_steps"], "monitoring.flush_every_steps"
            ),
        )


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"

    @classmethod
    def from_mapping(cls, value: Any) -> DistributedConfig:
        if value is None:
            return cls()
        data = _mapping(value, "distributed")
        _exact_keys(data, {"enabled", "backend"}, "distributed")
        if type(data["enabled"]) is not bool:
            raise TrainingConfigError("distributed.enabled must be a boolean")
        if data["backend"] not in {"nccl", "gloo"}:
            raise TrainingConfigError("distributed.backend must be 'nccl' or 'gloo'")
        return cls(enabled=data["enabled"], backend=data["backend"])


@dataclass(frozen=True, slots=True)
class TrainingBudgetConfig:
    stage: str
    expected_world_size: int
    expected_tokens_per_step: tuple[int, ...]
    expected_training_samples: int
    expected_total_tokens: int
    available_candidate_samples: int
    minimum_coverage_ratio: float

    @classmethod
    def from_mapping(cls, value: Any) -> TrainingBudgetConfig:
        data = _mapping(value, "budget")
        _exact_keys(
            data,
            {
                "stage",
                "expected_world_size",
                "expected_tokens_per_step",
                "expected_training_samples",
                "expected_total_tokens",
                "available_candidate_samples",
                "minimum_coverage_ratio",
            },
            "budget",
        )
        if data["stage"] not in {"A", "B", "C", "D", "R"}:
            raise TrainingConfigError("budget.stage must be A, B, C, D, or R")
        raw_tokens = data["expected_tokens_per_step"]
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise TrainingConfigError(
                "budget.expected_tokens_per_step must be a non-empty list"
            )
        return cls(
            stage=data["stage"],
            expected_world_size=_positive_int(
                data["expected_world_size"], "budget.expected_world_size"
            ),
            expected_tokens_per_step=tuple(
                _positive_int(value, f"budget.expected_tokens_per_step[{index}]")
                for index, value in enumerate(raw_tokens)
            ),
            expected_training_samples=_positive_int(
                data["expected_training_samples"],
                "budget.expected_training_samples",
            ),
            expected_total_tokens=_positive_int(
                data["expected_total_tokens"], "budget.expected_total_tokens"
            ),
            available_candidate_samples=_positive_int(
                data["available_candidate_samples"],
                "budget.available_candidate_samples",
            ),
            minimum_coverage_ratio=_finite_float(
                data["minimum_coverage_ratio"],
                "budget.minimum_coverage_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    source_stage: str
    source_config_path: Path
    source_config_sha256: str
    source_final_step: int
    load_optimizer_state: bool

    @classmethod
    def from_mapping(cls, value: Any) -> InitializationConfig:
        data = _mapping(value, "initialization")
        _exact_keys(
            data,
            {
                "source_stage",
                "source_config_path",
                "source_config_sha256",
                "source_final_step",
                "load_optimizer_state",
            },
            "initialization",
        )
        if data["source_stage"] not in {"A", "B", "C", "D"}:
            raise TrainingConfigError(
                "initialization.source_stage must be A, B, C, or D"
            )
        if type(data["load_optimizer_state"]) is not bool:
            raise TrainingConfigError(
                "initialization.load_optimizer_state must be a boolean"
            )
        return cls(
            source_stage=data["source_stage"],
            source_config_path=_relative_path(
                data["source_config_path"], "initialization.source_config_path"
            ),
            source_config_sha256=_sha256(
                data["source_config_sha256"],
                "initialization.source_config_sha256",
            ),
            source_final_step=_positive_int(
                data["source_final_step"], "initialization.source_final_step"
            ),
            load_optimizer_state=data["load_optimizer_state"],
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    schema_version: int
    name: str
    status: str
    seed: int
    model: ModelBinding
    data: DataBinding
    batch: BatchConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    stability: StabilityConfig
    checkpoint: CheckpointConfig
    runtime: RuntimeConfig
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    budget: TrainingBudgetConfig | None = None
    initialization: InitializationConfig | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> TrainingConfig:
        data = _mapping(value, "training config")
        required = {
            "schema_version",
            "name",
            "status",
            "seed",
            "model",
            "data",
            "batch",
            "optimizer",
            "scheduler",
            "stability",
            "checkpoint",
            "runtime",
        }
        optional = {"monitoring", "distributed", "budget", "initialization"}
        unknown = set(data) - required - optional
        missing = required - set(data)
        if missing or unknown:
            _exact_keys(data, required | (optional & set(data)), "training config")
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != TRAINING_SCHEMA_VERSION
        ):
            raise TrainingConfigError(
                f"schema_version must be {TRAINING_SCHEMA_VERSION}"
            )
        if not isinstance(data["status"], str) or data["status"] not in {
            "smoke",
            "release",
        }:
            raise TrainingConfigError("status must be 'smoke' or 'release'")
        seed = _non_negative_int(data["seed"], "seed")
        model = ModelBinding.from_mapping(data["model"])
        data_binding = DataBinding.from_mapping(data["data"])
        batch = BatchConfig.from_mapping(data["batch"])
        scheduler = SchedulerConfig.from_mapping(data["scheduler"])
        checkpoint = CheckpointConfig.from_mapping(data["checkpoint"])
        if data["status"] == "release" and not data_binding.formal_training_eligible:
            raise TrainingConfigError(
                "release training requires formally eligible data"
            )
        budget = (
            None
            if data.get("budget") is None
            else TrainingBudgetConfig.from_mapping(data["budget"])
        )
        initialization = (
            None
            if data.get("initialization") is None
            else InitializationConfig.from_mapping(data["initialization"])
        )
        config = cls(
            schema_version=TRAINING_SCHEMA_VERSION,
            name=_name(data["name"], "name"),
            status=data["status"],
            seed=seed,
            model=model,
            data=data_binding,
            batch=batch,
            optimizer=OptimizerConfig.from_mapping(data["optimizer"]),
            scheduler=scheduler,
            stability=StabilityConfig.from_mapping(data["stability"]),
            checkpoint=checkpoint,
            runtime=RuntimeConfig.from_mapping(data["runtime"]),
            monitoring=MonitoringConfig.from_mapping(data.get("monitoring")),
            distributed=DistributedConfig.from_mapping(data.get("distributed")),
            budget=budget,
            initialization=initialization,
        )
        if budget is not None:
            schedule = batch.gradient_accumulation_schedule or (
                batch.gradient_accumulation_steps,
            )
            expected_tokens = tuple(
                batch.tokens_per_micro_batch * steps * budget.expected_world_size
                for steps in schedule
            )
            if budget.expected_tokens_per_step != expected_tokens:
                raise TrainingConfigError(
                    "budget.expected_tokens_per_step does not match batch schedule"
                )
            actual_samples = batch.samples_through(
                scheduler.total_steps,
                budget.expected_world_size,
            )
            actual_tokens = batch.tokens_through(
                scheduler.total_steps,
                budget.expected_world_size,
            )
            if budget.expected_training_samples != actual_samples:
                raise TrainingConfigError(
                    "budget.expected_training_samples does not match schedule"
                )
            if budget.expected_total_tokens != actual_tokens:
                raise TrainingConfigError(
                    "budget.expected_total_tokens does not match schedule"
                )
            coverage = actual_samples / budget.available_candidate_samples
            if coverage < budget.minimum_coverage_ratio:
                raise TrainingConfigError("budget coverage is below its minimum")
            required_source = {
                "A": None,
                "B": "A",
                "C": "B",
                "D": "C",
                "R": "D",
            }[budget.stage]
            actual_source = (
                None if initialization is None else initialization.source_stage
            )
            if actual_source != required_source:
                raise TrainingConfigError(
                    f"stage {budget.stage} initialization source must be "
                    f"{required_source}"
                )
        return config


@dataclass(frozen=True, slots=True)
class ExperimentTrial:
    name: str
    scheduler: str
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_gradient_norm: float
    micro_batch_size: int
    gradient_accumulation_steps: int

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> ExperimentTrial:
        context = f"trials[{index}]"
        data = _mapping(value, context)
        _exact_keys(
            data,
            {
                "name",
                "scheduler",
                "learning_rate",
                "warmup_steps",
                "weight_decay",
                "max_gradient_norm",
                "micro_batch_size",
                "gradient_accumulation_steps",
            },
            context,
        )
        if not isinstance(data["scheduler"], str) or data["scheduler"] not in {
            "cosine",
            "constant",
        }:
            raise TrainingConfigError(
                f"{context}.scheduler must be 'cosine' or 'constant'"
            )
        return cls(
            name=_name(data["name"], f"{context}.name"),
            scheduler=data["scheduler"],
            learning_rate=_finite_float(
                data["learning_rate"],
                f"{context}.learning_rate",
                minimum=1e-12,
            ),
            warmup_steps=_non_negative_int(
                data["warmup_steps"],
                f"{context}.warmup_steps",
            ),
            weight_decay=_finite_float(
                data["weight_decay"],
                f"{context}.weight_decay",
                minimum=0,
            ),
            max_gradient_norm=_finite_float(
                data["max_gradient_norm"],
                f"{context}.max_gradient_norm",
                minimum=1e-12,
            ),
            micro_batch_size=_positive_int(
                data["micro_batch_size"],
                f"{context}.micro_batch_size",
            ),
            gradient_accumulation_steps=_positive_int(
                data["gradient_accumulation_steps"],
                f"{context}.gradient_accumulation_steps",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    schema_version: int
    name: str
    status: str
    base_config_path: Path
    base_config_sha256: str
    trials: tuple[ExperimentTrial, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> ExperimentMatrix:
        data = _mapping(value, "experiment matrix")
        _exact_keys(
            data,
            {
                "schema_version",
                "name",
                "status",
                "base_config_path",
                "base_config_sha256",
                "trials",
            },
            "experiment matrix",
        )
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != MATRIX_SCHEMA_VERSION
        ):
            raise TrainingConfigError(f"schema_version must be {MATRIX_SCHEMA_VERSION}")
        if not isinstance(data["status"], str) or data["status"] != "smoke":
            raise TrainingConfigError("experiment matrix status must be 'smoke'")
        if not isinstance(data["trials"], list) or not data["trials"]:
            raise TrainingConfigError("trials must be a non-empty list")
        trials = tuple(
            ExperimentTrial.from_mapping(trial, index)
            for index, trial in enumerate(data["trials"])
        )
        trial_names = [trial.name for trial in trials]
        if len(set(trial_names)) != len(trial_names):
            raise TrainingConfigError("trial names must be unique")
        if "baseline" not in trial_names:
            raise TrainingConfigError("trials must contain 'baseline'")
        return cls(
            schema_version=MATRIX_SCHEMA_VERSION,
            name=_name(data["name"], "name"),
            status="smoke",
            base_config_path=_relative_path(
                data["base_config_path"],
                "base_config_path",
            ),
            base_config_sha256=_sha256(
                data["base_config_sha256"],
                "base_config_sha256",
            ),
            trials=trials,
        )


def _read_yaml(path: Path, context: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{context} not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise TrainingConfigError(f"invalid YAML in {path}: {error}") from error


def load_training_config(
    path: str | Path,
    *,
    project_root: str | Path = ".",
) -> TrainingConfig:
    config_path = Path(path)
    config = TrainingConfig.from_mapping(_read_yaml(config_path, "training config"))
    model_path = Path(project_root) / config.model.config_path
    if file_sha256(model_path) != config.model.config_sha256:
        raise TrainingConfigError("model config SHA-256 does not match")
    model_config = load_model_config(model_path)
    if (
        model_config.components.attention_dropout != 0.0
        or model_config.components.residual_dropout != 0.0
    ):
        raise TrainingConfigError(
            "base pretraining requires zero attention and residual dropout"
        )
    if model_config.name != config.model.name:
        raise TrainingConfigError("bound model name does not match model config")
    if model_config.expected_parameter_count != config.model.expected_parameter_count:
        raise TrainingConfigError("bound parameter count does not match model config")
    if model_config.tokenizer.version_id != config.data.tokenizer_version_id:
        raise TrainingConfigError("bound tokenizer version does not match model config")
    if model_config.tokenizer.tokenizer_sha256 != config.data.tokenizer_sha256:
        raise TrainingConfigError("bound tokenizer SHA-256 does not match model config")
    if config.batch.sequence_length > model_config.dimensions.max_sequence_length:
        raise TrainingConfigError("batch sequence length exceeds model context")
    if config.initialization is not None:
        source_path = Path(project_root) / config.initialization.source_config_path
        if file_sha256(source_path) != config.initialization.source_config_sha256:
            raise TrainingConfigError("initialization source config SHA-256 mismatch")
    return config


def load_experiment_matrix(
    path: str | Path,
    *,
    project_root: str | Path = ".",
) -> tuple[ExperimentMatrix, TrainingConfig]:
    matrix_path = Path(path)
    matrix = ExperimentMatrix.from_mapping(_read_yaml(matrix_path, "experiment matrix"))
    base_path = Path(project_root) / matrix.base_config_path
    if file_sha256(base_path) != matrix.base_config_sha256:
        raise TrainingConfigError("base training config SHA-256 does not match")
    base_config = load_training_config(base_path, project_root=project_root)
    for trial in matrix.trials:
        if trial.warmup_steps >= base_config.scheduler.total_steps:
            raise TrainingConfigError(
                f"trial {trial.name!r} warmup must be less than total steps"
            )
    return matrix, base_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a stage-4 training baseline or experiment matrix."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/atom-50m-baseline.yaml"),
    )
    selection.add_argument(
        "--matrix",
        type=Path,
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.matrix is None:
        base = load_training_config(args.config, project_root=args.project_root)
        summary: dict[str, Any] = {
            "config": base.name,
            "formal_training_eligible": base.data.formal_training_eligible,
            "model": base.model.name,
            "parameter_count": base.model.expected_parameter_count,
            "total_steps": base.scheduler.total_steps,
            "tokens_per_rank_at_max_accumulation": (
                base.batch.tokens_per_optimizer_step
            ),
        }
        if base.budget is not None:
            summary.update(
                {
                    "stage": base.budget.stage,
                    "dataset_id": base.data.data_version_id,
                    "dataset_manifest_sha256": (base.data.dataset_manifest_sha256),
                    "world_size": base.budget.expected_world_size,
                    "sequence_length": base.batch.sequence_length,
                    "micro_batch_size": base.batch.micro_batch_size,
                    "gradient_accumulation_schedule": (
                        list(base.batch.gradient_accumulation_schedule)
                        or [base.batch.gradient_accumulation_steps]
                    ),
                    "global_tokens_per_step": list(
                        base.budget.expected_tokens_per_step
                    ),
                    "expected_total_tokens": base.budget.expected_total_tokens,
                    "expected_training_samples": (
                        base.budget.expected_training_samples
                    ),
                    "available_candidate_samples": (
                        base.budget.available_candidate_samples
                    ),
                    "coverage_ratio": (
                        base.budget.expected_training_samples
                        / base.budget.available_candidate_samples
                    ),
                }
            )
        print(json.dumps(summary, sort_keys=True))
        return 0
    matrix, base = load_experiment_matrix(args.matrix, project_root=args.project_root)
    summary = {
        "base_config": base.name,
        "formal_training_eligible": base.data.formal_training_eligible,
        "matrix": matrix.name,
        "model": base.model.name,
        "parameter_count": base.model.expected_parameter_count,
        "total_steps_per_trial": base.scheduler.total_steps,
        "tokens_per_optimizer_step": base.batch.tokens_per_optimizer_step,
        "trial_count": len(matrix.trials),
        "trials": [trial.name for trial in matrix.trials],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
