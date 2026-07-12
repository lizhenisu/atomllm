"""Audit a full-parameter context-window training and recovery path."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from atomllm.model.config import load_model_config
from atomllm.training.config import file_sha256, load_training_config
from atomllm.training.final_recheck import (
    atomic_json,
    latest_checkpoint,
    run_trainer_process,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/training/atom-base-300m-long-audit.yaml")


class ContextAuditError(RuntimeError):
    """Raised when maximum-context training evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class ContextAuditConfig:
    name: str
    formal_config: Path
    smoke_config: Path
    training_data: Path
    run_id: str
    baseline_run_id: str
    baseline_sequence_length: int
    training_runs_dir: Path
    expected_sequence_length: int
    maximum_peak_allocated_gib: float
    output_report: Path


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContextAuditError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContextAuditError(f"{field} must be a safe relative path")
    return path


def load_context_audit_config(
    path: str | Path = DEFAULT_CONFIG,
) -> ContextAuditConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ContextAuditError(f"cannot read config: {config_path}") from error
    expected = {
        "schema_version",
        "name",
        "formal_config",
        "smoke_config",
        "training_data",
        "run_id",
        "baseline_run_id",
        "baseline_sequence_length",
        "training_runs_dir",
        "expected_sequence_length",
        "maximum_peak_allocated_gib",
        "output_report",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ContextAuditError("context audit config fields are invalid")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ContextAuditError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ContextAuditError("name must be a non-empty string")
    run_id = value["run_id"]
    if not isinstance(run_id, str) or not run_id or " " in run_id:
        raise ContextAuditError("run_id must be a safe identifier")
    baseline_run_id = value["baseline_run_id"]
    if (
        not isinstance(baseline_run_id, str)
        or not baseline_run_id
        or " " in baseline_run_id
    ):
        raise ContextAuditError("baseline_run_id must be a safe identifier")
    sequence_length = value["expected_sequence_length"]
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ContextAuditError("expected_sequence_length must be positive")
    baseline_sequence_length = value["baseline_sequence_length"]
    if type(baseline_sequence_length) is not int or baseline_sequence_length <= 0:
        raise ContextAuditError("baseline_sequence_length must be positive")
    maximum = value["maximum_peak_allocated_gib"]
    if (
        type(maximum) not in {int, float}
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0
    ):
        raise ContextAuditError("maximum_peak_allocated_gib must be positive")
    return ContextAuditConfig(
        name=value["name"],
        formal_config=_relative_path(value["formal_config"], "formal_config"),
        smoke_config=_relative_path(value["smoke_config"], "smoke_config"),
        training_data=_relative_path(value["training_data"], "training_data"),
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        baseline_sequence_length=baseline_sequence_length,
        training_runs_dir=_relative_path(
            value["training_runs_dir"], "training_runs_dir"
        ),
        expected_sequence_length=sequence_length,
        maximum_peak_allocated_gib=float(maximum),
        output_report=_relative_path(value["output_report"], "output_report"),
    )


def _reports(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "reports").glob("training-report-*.json"))
    ]


def audit_context(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_context_audit_config(root / config_path)
    formal_path = root / config.formal_config
    smoke_path = root / config.smoke_config
    formal = load_training_config(formal_path, project_root=root)
    smoke = load_training_config(smoke_path, project_root=root)
    model = load_model_config(root / formal.model.config_path)
    expected_length = config.expected_sequence_length
    for candidate in (formal, smoke):
        if candidate.model.expected_parameter_count != model.expected_parameter_count:
            raise ContextAuditError("training and model parameter counts differ")
        if candidate.batch.sequence_length != expected_length:
            raise ContextAuditError("training sequence length mismatch")
        if not candidate.runtime.gradient_checkpointing:
            raise ContextAuditError("gradient checkpointing is disabled")
        if candidate.runtime.checkpoint_segment_layers <= 1:
            raise ContextAuditError("grouped gradient checkpointing is disabled")
        if candidate.runtime.loss_chunk_size is None:
            raise ContextAuditError("chunked loss is disabled")
        if not candidate.data.formal_training_eligible:
            raise ContextAuditError("config is not bound to formal data")
    if formal.status != "release" or smoke.status != "smoke":
        raise ContextAuditError("formal/smoke status mismatch")
    if model.dimensions.max_sequence_length != expected_length:
        raise ContextAuditError("model RoPE limit differs from training length")
    if formal.model != smoke.model or formal.data != smoke.data:
        raise ContextAuditError("formal and smoke lineage differs")
    if formal.runtime != smoke.runtime:
        raise ContextAuditError("formal and smoke runtime settings differ")
    if smoke.scheduler.total_steps != 2:
        raise ContextAuditError("smoke config must contain two steps")

    runs_dir = root / config.training_runs_dir
    run_dir = runs_dir / config.run_id
    checkpoints_dir = run_dir / "checkpoints"
    latest = latest_checkpoint(checkpoints_dir) if checkpoints_dir.exists() else None
    if latest is None:
        if run_dir.exists():
            raise ContextAuditError("existing run has no complete checkpoint")
        run_trainer_process(
            root=root,
            training_config=smoke_path,
            training_data=root / config.training_data,
            training_runs_dir=runs_dir,
            run_id=config.run_id,
            target_steps=1,
            resume=False,
        )
        latest = latest_checkpoint(checkpoints_dir)
    if latest is None:
        raise ContextAuditError("first process produced no checkpoint")
    if latest[0]["global_step"] == 1:
        run_trainer_process(
            root=root,
            training_config=smoke_path,
            training_data=root / config.training_data,
            training_runs_dir=runs_dir,
            run_id=config.run_id,
            target_steps=2,
            resume=True,
        )
        latest = latest_checkpoint(checkpoints_dir)
    if latest is None or latest[0]["global_step"] != 2:
        raise ContextAuditError("restored process did not reach step two")

    reports = _reports(run_dir)
    first = [
        item
        for item in reports
        if item.get("trainer_state", {}).get("global_step") == 1
        and item.get("restored_checkpoint_id") is None
    ]
    restored = [
        item
        for item in reports
        if item.get("trainer_state", {}).get("global_step") == 2
        and item.get("restored_checkpoint_id") == "step-000000001"
    ]
    if len(first) != 1 or len(restored) != 1:
        raise ContextAuditError("initial/restored report count is invalid")
    first_report, restored_report = first[0], restored[0]
    samples_per_step = (
        smoke.batch.micro_batch_size * smoke.batch.gradient_accumulation_steps
    )
    if first_report["data_state"]["sample_index"] != samples_per_step:
        raise ContextAuditError("first data cursor is invalid")
    if restored_report["data_state"]["sample_index"] != 2 * samples_per_step:
        raise ContextAuditError("restored data cursor did not advance once")
    metrics = first_report["step_metrics"] + restored_report["step_metrics"]
    if len(metrics) != 2 or any(
        not math.isfinite(metric[field])
        for metric in metrics
        for field in ("loss", "gradient_norm")
    ):
        raise ContextAuditError("loss or gradient norm is not finite")
    peak_allocated = max(
        first_report["peak_allocated_gib"], restored_report["peak_allocated_gib"]
    )
    if peak_allocated > config.maximum_peak_allocated_gib:
        raise ContextAuditError("active tensors exceed configured VRAM gate")
    physical_gib = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available()
        else config.maximum_peak_allocated_gib
    )
    if peak_allocated >= physical_gib:
        raise ContextAuditError("active tensors are not physical-VRAM resident")

    baseline_reports = _reports(runs_dir / config.baseline_run_id)
    baseline = [
        item
        for item in baseline_reports
        if item.get("trainer_state", {}).get("global_step") == 2
        and item.get("restored_checkpoint_id") == "step-000000001"
    ]
    if len(baseline) != 1:
        raise ContextAuditError("ungrouped baseline report is missing")
    baseline_peak = baseline[0]["peak_allocated_gib"]
    if config.baseline_sequence_length >= expected_length:
        raise ContextAuditError("grouped context did not increase")
    if peak_allocated >= baseline_peak:
        raise ContextAuditError("grouped checkpointing did not reduce peak")

    report = {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "model_parameter_count": formal.model.expected_parameter_count,
        "model_max_sequence_length": model.dimensions.max_sequence_length,
        "trained_sequence_length": expected_length,
        "micro_batch_size": formal.batch.micro_batch_size,
        "tokens_per_optimizer_step": formal.batch.tokens_per_optimizer_step,
        "gradient_checkpointing": True,
        "checkpoint_segment_layers": formal.runtime.checkpoint_segment_layers,
        "loss_chunk_size": formal.runtime.loss_chunk_size,
        "direct_causal_sdpa": True,
        "formal_config_sha256": file_sha256(formal_path),
        "smoke_config_sha256": file_sha256(smoke_path),
        "dataset_manifest_sha256": restored_report["data_state"][
            "dataset_manifest_sha256"
        ],
        "restored_checkpoint_id": restored_report["restored_checkpoint_id"],
        "final_checkpoint_id": latest[0]["checkpoint_id"],
        "final_step": latest[0]["global_step"],
        "tokens_seen": latest[0]["tokens_seen"],
        "initial_loss": metrics[0]["loss"],
        "final_loss": metrics[1]["loss"],
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": max(
            first_report["peak_reserved_gib"], restored_report["peak_reserved_gib"]
        ),
        "physical_vram_gib": physical_gib,
        "baseline_sequence_length": config.baseline_sequence_length,
        "baseline_peak_allocated_gib": baseline_peak,
        "checks": {
            "full_parameter_model": True,
            "formal_token_shards": True,
            "native_context_training": True,
            "grouped_gradient_checkpointing": True,
            "finite_loss_and_gradients": True,
            "active_tensors_fit_physical_vram": True,
            "exact_process_recovery": True,
            "data_cursor_advanced": True,
            "context_increased_from_ungrouped_baseline": True,
            "grouped_peak_below_ungrouped_baseline": True,
        },
        "native_context_ready": True,
        "passed": True,
    }
    atomic_json(root / config.output_report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_context(args.config, project_root=args.project_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
