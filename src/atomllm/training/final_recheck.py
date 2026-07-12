"""Run the selected Atom-50M candidate across a real process restart."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.checkpoint import verify_checkpoint_directory
from atomllm.training.config import file_sha256, load_training_config
from atomllm.training.data import ShardedTokenDataset
from atomllm.training.sequential_search import (
    evaluate_validation_loss,
    load_sequential_search_plan,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/training/atom-50m-final-recheck.yaml")


class FinalRecheckError(RuntimeError):
    """Raised when final-candidate process recovery cannot be proven."""


@dataclass(frozen=True, slots=True)
class FinalRecheckConfig:
    name: str
    search_plan: Path
    run_id: str
    training_runs_dir: Path
    output_report: Path


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FinalRecheckError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FinalRecheckError(f"{field} must be a safe relative path")
    return path


def load_final_recheck_config(
    path: str | Path = DEFAULT_CONFIG,
) -> FinalRecheckConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FinalRecheckError(f"cannot read config: {config_path}") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "name",
        "search_plan",
        "run_id",
        "training_runs_dir",
        "output_report",
    }:
        raise FinalRecheckError("final recheck config fields are invalid")
    if value["schema_version"] != SCHEMA_VERSION:
        raise FinalRecheckError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise FinalRecheckError("name must be a non-empty string")
    run_id = value["run_id"]
    if (
        not isinstance(run_id, str)
        or not run_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in run_id
        )
    ):
        raise FinalRecheckError("run_id contains unsupported characters")
    return FinalRecheckConfig(
        name=value["name"],
        search_plan=_safe_path(value["search_plan"], "search_plan"),
        run_id=run_id,
        training_runs_dir=_safe_path(value["training_runs_dir"], "training_runs_dir"),
        output_report=_safe_path(value["output_report"], "output_report"),
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def latest_checkpoint(checkpoints_dir: Path) -> tuple[dict[str, Any], Path] | None:
    latest_path = checkpoints_dir / "latest.json"
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_dir = checkpoints_dir / latest["checkpoint_id"]
    manifest = verify_checkpoint_directory(
        checkpoint_dir,
        expected_manifest_sha256=latest["manifest_sha256"],
    )
    return manifest, checkpoint_dir


def run_trainer_process(
    *,
    root: Path,
    training_config: Path,
    training_data: Path,
    training_runs_dir: Path,
    run_id: str,
    target_steps: int,
    resume: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "atomllm.training.trainer",
        "--config",
        str(training_config.relative_to(root)),
        "--training-data",
        str(training_data.relative_to(root)),
        "--steps",
        str(target_steps),
        "--output-dir",
        str(training_runs_dir.relative_to(root)),
        "--run-id",
        run_id,
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=root, check=True)


def find_restored_report(reports_dir: Path, expected_final_step: int) -> dict[str, Any]:
    reports = []
    for path in reports_dir.glob("training-report-*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("trainer_state", {}).get("global_step") == expected_final_step:
            reports.append(value)
    restored = [report for report in reports if report.get("restored_checkpoint_id")]
    if len(restored) != 1:
        raise FinalRecheckError(
            "exactly one restored final training report is required"
        )
    return restored[0]


def run_final_recheck(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_final_recheck_config(root / config_path)
    plan = load_sequential_search_plan(root / config.search_plan)
    search_output = root / plan.output_dir
    summary_path = search_output / "summary.json"
    if not summary_path.is_file():
        raise FinalRecheckError("sequential search summary is missing")
    search_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_config_path = root / search_summary["final_config"]
    training_config = load_training_config(training_config_path, project_root=root)
    total_steps = training_config.scheduler.total_steps
    if total_steps % 2:
        raise FinalRecheckError("final candidate total steps must be even")
    midpoint = total_steps // 2
    training_data = root / plan.training_data
    runs_dir = root / config.training_runs_dir
    run_dir = runs_dir / config.run_id
    checkpoints_dir = run_dir / "checkpoints"
    latest = latest_checkpoint(checkpoints_dir) if checkpoints_dir.exists() else None
    if latest is None:
        if run_dir.exists():
            raise FinalRecheckError(
                "existing final run has no complete checkpoint; choose a clean run_id"
            )
        run_trainer_process(
            root=root,
            training_config=training_config_path,
            training_data=training_data,
            training_runs_dir=runs_dir,
            run_id=config.run_id,
            target_steps=midpoint,
            resume=False,
        )
        latest = latest_checkpoint(checkpoints_dir)
    if latest is None:
        raise FinalRecheckError("midpoint process produced no checkpoint")
    midpoint_manifest, _ = latest
    if midpoint_manifest["global_step"] < midpoint:
        run_trainer_process(
            root=root,
            training_config=training_config_path,
            training_data=training_data,
            training_runs_dir=runs_dir,
            run_id=config.run_id,
            target_steps=midpoint,
            resume=True,
        )
        latest = latest_checkpoint(checkpoints_dir)
        if latest is None:
            raise FinalRecheckError("resumed midpoint process produced no checkpoint")
        midpoint_manifest, _ = latest
    if midpoint_manifest["global_step"] == midpoint:
        run_trainer_process(
            root=root,
            training_config=training_config_path,
            training_data=training_data,
            training_runs_dir=runs_dir,
            run_id=config.run_id,
            target_steps=total_steps,
            resume=True,
        )
    latest = latest_checkpoint(checkpoints_dir)
    if latest is None:
        raise FinalRecheckError("final process produced no checkpoint")
    final_manifest, final_checkpoint_dir = latest
    if final_manifest["global_step"] != total_steps:
        raise FinalRecheckError("final checkpoint did not reach configured total steps")
    restored_report = find_restored_report(run_dir / "reports", total_steps)
    if restored_report["restored_global_step"] != midpoint:
        raise FinalRecheckError("final process did not restore the midpoint checkpoint")

    model_config = load_model_config(root / training_config.model.config_path)
    model = AtomLLM(model_config).to(torch.device(training_config.runtime.device))
    load_safetensors_checkpoint(model, final_checkpoint_dir / "model.safetensors")
    validation_dataset = ShardedTokenDataset(
        root / plan.validation_data,
        sequence_length=training_config.batch.sequence_length,
    )
    validation_loss = evaluate_validation_loss(
        model,
        validation_dataset,
        training_config,
        batches=plan.validation_batches,
        batch_size=plan.validation_batch_size,
        seed=plan.validation_seed,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "search_summary_sha256": file_sha256(summary_path),
        "training_config": str(training_config_path.relative_to(root)),
        "training_config_sha256": file_sha256(training_config_path),
        "run_id": config.run_id,
        "midpoint_step": midpoint,
        "restored_checkpoint_id": restored_report["restored_checkpoint_id"],
        "final_checkpoint_id": final_manifest["checkpoint_id"],
        "final_step": final_manifest["global_step"],
        "tokens_seen": final_manifest["tokens_seen"],
        "validation_loss": validation_loss,
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
    report = run_final_recheck(args.config, project_root=args.project_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
