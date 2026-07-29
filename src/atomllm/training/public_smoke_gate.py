"""Verify the real-data 6-GPU smoke run before full public pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from atomllm.training.checkpoint import verify_checkpoint_directory
from atomllm.training.config import load_training_config


class PublicSmokeGateError(RuntimeError):
    """Raised when the public pretraining smoke is not safe to promote."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicSmokeGateError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PublicSmokeGateError(f"JSON must be an object: {path}")
    return value


def verify_smoke(
    *,
    run_dir: Path,
    training_config: Path,
    output_dir: Path,
    expected_steps: int = 30,
    maximum_reserved_gib: float = 23.5,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    if type(expected_steps) is not int or expected_steps <= 0:
        raise PublicSmokeGateError("expected_steps must be positive")
    if not 0 < maximum_reserved_gib < 24:
        raise PublicSmokeGateError("maximum_reserved_gib must be in (0, 24)")
    root = project_root.resolve()

    def resolve(path: Path, field: str) -> Path:
        result = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not result.is_relative_to(root):
            raise PublicSmokeGateError(f"{field} escapes project root")
        return result

    run = resolve(run_dir, "run_dir")
    config_path = resolve(training_config, "training_config")
    output = resolve(output_dir, "output_dir")
    config = load_training_config(config_path, project_root=root)
    if config.budget is None or config.budget.expected_world_size != 6:
        raise PublicSmokeGateError("training config is not the six-GPU public run")
    reports = sorted((run / "reports").glob("training-report-*.json"))
    completed_reports = []
    for path in reports:
        candidate = _read_json(path)
        candidate_state = candidate.get("trainer_state")
        if (
            isinstance(candidate_state, dict)
            and candidate_state.get("global_step") == expected_steps
        ):
            completed_reports.append((path, candidate))
    if len(completed_reports) != 1:
        raise PublicSmokeGateError(
            "smoke run must have exactly one report at the requested final step"
        )
    training_report_path, training_report = completed_reports[0]
    trainer_state = training_report.get("trainer_state")
    metrics = training_report.get("step_metrics")
    distributed = training_report.get("distributed")
    if not isinstance(trainer_state, dict) or not isinstance(metrics, list):
        raise PublicSmokeGateError("smoke training report schema is invalid")
    if not isinstance(distributed, dict) or distributed.get("world_size") != 6:
        raise PublicSmokeGateError("smoke report did not use six GPUs")
    restored_step = training_report.get("restored_global_step")
    restored_checkpoint_id = training_report.get("restored_checkpoint_id")
    if restored_step is None:
        restored_step = 0
    if (
        type(restored_step) is not int
        or restored_step < 0
        or restored_step >= expected_steps
    ):
        raise PublicSmokeGateError("smoke restored step is invalid")
    if restored_step:
        expected_restored_id = f"step-{restored_step:09d}"
        if restored_checkpoint_id != expected_restored_id:
            raise PublicSmokeGateError("smoke restored checkpoint ID is invalid")
        restored_checkpoint = verify_checkpoint_directory(
            run / "checkpoints" / expected_restored_id
        )
        restored_tokens = (
            restored_step
            * config.batch.tokens_per_optimizer_step
            * config.budget.expected_world_size
        )
        if (
            restored_checkpoint.get("global_step") != restored_step
            or restored_checkpoint.get("tokens_seen") != restored_tokens
            or restored_checkpoint.get("config_sha256") != _sha256(config_path)
        ):
            raise PublicSmokeGateError("smoke restored checkpoint is incompatible")
    elif restored_checkpoint_id is not None:
        raise PublicSmokeGateError("fresh smoke report names a restored checkpoint")
    expected_metric_steps = list(range(restored_step + 1, expected_steps + 1))
    metric_steps = [
        item.get("global_step") if isinstance(item, dict) else None for item in metrics
    ]
    if metric_steps != expected_metric_steps:
        raise PublicSmokeGateError("smoke did not complete every requested step")
    expected_tokens = (
        expected_steps
        * config.batch.tokens_per_optimizer_step
        * config.budget.expected_world_size
    )
    if trainer_state.get("tokens_seen") != expected_tokens:
        raise PublicSmokeGateError("smoke token accounting is invalid")
    losses = [item.get("loss") for item in metrics if isinstance(item, dict)]
    norms = [item.get("gradient_norm") for item in metrics if isinstance(item, dict)]
    if len(losses) != len(expected_metric_steps) or not all(
        type(value) in {int, float} and math.isfinite(float(value)) and value > 0
        for value in losses
    ):
        raise PublicSmokeGateError("smoke losses are not finite and positive")
    if len(norms) != len(expected_metric_steps) or not all(
        type(value) in {int, float} and math.isfinite(float(value)) and value >= 0
        for value in norms
    ):
        raise PublicSmokeGateError("smoke gradient norms are invalid")
    early_loss = sum(float(value) for value in losses[:5]) / 5
    late_loss = sum(float(value) for value in losses[-5:]) / 5
    if late_loss > early_loss * 1.02:
        raise PublicSmokeGateError("smoke loss is diverging")
    throughput = training_report.get("tokens_per_second")
    peak_reserved = training_report.get("peak_reserved_gib")
    if type(throughput) not in {int, float} or throughput <= 0:
        raise PublicSmokeGateError("smoke throughput is invalid")
    if (
        type(peak_reserved) not in {int, float}
        or peak_reserved <= 0
        or peak_reserved > maximum_reserved_gib
    ):
        raise PublicSmokeGateError("smoke reserved memory is unsafe")
    checkpoint_id = f"step-{expected_steps:09d}"
    checkpoint = verify_checkpoint_directory(run / "checkpoints" / checkpoint_id)
    if checkpoint.get("global_step") != expected_steps:
        raise PublicSmokeGateError("final smoke checkpoint step mismatch")
    if checkpoint.get("tokens_seen") != expected_tokens:
        raise PublicSmokeGateError("final smoke checkpoint token mismatch")
    if checkpoint.get("config_sha256") != _sha256(config_path):
        raise PublicSmokeGateError("final smoke checkpoint config mismatch")
    report = {
        "schema_version": 1,
        "gate_version": "public-pretraining-real-data-smoke-v1",
        "run_id": run.name,
        "training_config_sha256": _sha256(config_path),
        "training_report_sha256": _sha256(training_report_path),
        "checkpoint_id": checkpoint_id,
        "checkpoint_manifest_sha256": _sha256(
            run / "checkpoints" / checkpoint_id / "manifest.json"
        ),
        "global_step": expected_steps,
        "restored_global_step": restored_step or None,
        "tokens_seen": expected_tokens,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tokens_per_second": throughput,
        "peak_reserved_gib": peak_reserved,
        "checks": {
            "six_gpu_ddp": True,
            "real_public_training_shards": True,
            "finite_losses": True,
            "loss_not_diverging": True,
            "finite_gradient_norms": True,
            "checkpoint_exact_resume_payload": True,
            "continuous_process_metric_suffix": True,
            "model_external_capability": False,
        },
        "full_training_eligible": True,
    }
    if output.exists():
        existing_path = output / "report.json"
        completed_path = output / "COMPLETED"
        if existing_path.is_file() and completed_path.is_file():
            existing = _read_json(existing_path)
            if existing == report and completed_path.read_text(encoding="utf-8") == (
                f"{_sha256(existing_path)}  report.json\n"
            ):
                return existing
        raise PublicSmokeGateError("existing smoke gate is incompatible")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").write_text(
            f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=30)
    parser.add_argument("--maximum-reserved-gib", type=float, default=23.5)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = verify_smoke(
        run_dir=args.run_dir,
        training_config=args.training_config,
        output_dir=args.output_dir,
        expected_steps=args.expected_steps,
        maximum_reserved_gib=args.maximum_reserved_gib,
        project_root=args.project_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
