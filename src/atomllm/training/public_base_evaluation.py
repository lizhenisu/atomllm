"""Run the immutable six-GPU delivery gate for the public 100B base model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.tokenizer.evaluation import verify_tokenizer_directory
from atomllm.training.base_benchmark import evaluate
from atomllm.training.config import DistributedConfig
from atomllm.training.distributed import DistributedContext
from atomllm.training.public_token_shards import tokenizer_from_gpu_selection


EXPECTED_SUITE_ID = "atom-base-public-zero-shot-full-v3"
EXPECTED_BENCHMARK_COUNTS = {
    "arc_challenge": 1172,
    "ceval": 1346,
    "hellaswag": 10042,
    "mmlu": 14042,
}
EXPECTED_WORLD_SIZE = 6


class PublicBaseEvaluationError(RuntimeError):
    """Raised when final public base-model evidence is incomplete or mutable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicBaseEvaluationError(f"JSON object required: {path}")
    return value


def _completed(directory: Path, payload_name: str) -> tuple[dict[str, Any], str]:
    payload = directory / payload_name
    marker = directory / "COMPLETED"
    if not payload.is_file() or not marker.is_file():
        raise PublicBaseEvaluationError(f"artifact is incomplete: {directory}")
    payload_sha = _sha256(payload)
    if marker.read_text(encoding="utf-8") != f"{payload_sha}  {payload_name}\n":
        raise PublicBaseEvaluationError(f"artifact marker is invalid: {directory}")
    return _json(payload), payload_sha


def resolve_contract(
    *,
    run_dir: Path,
    release_dir: Path,
    tokenizer_selection_dir: Path,
    suite_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    paths = {
        "run": run_dir,
        "release": release_dir,
        "selection": tokenizer_selection_dir,
        "suite": suite_dir,
    }
    resolved = {
        name: (root / path).resolve() if not path.is_absolute() else path.resolve()
        for name, path in paths.items()
    }
    if any(not path.is_relative_to(root) for path in resolved.values()):
        raise PublicBaseEvaluationError("evaluation paths must stay in project root")

    release, release_sha = _completed(resolved["release"], "manifest.json")
    if (
        release.get("training_eligible") is not True
        or release.get("checks", {}).get("full_public_100b_plan") is not True
        or release.get("checks", {}).get("synthetic_training_content") is not False
        or release.get("checks", {}).get("model_external_capability") is not False
    ):
        raise PublicBaseEvaluationError("training release is not delivery eligible")
    training_path = resolved["release"] / release["training_config"]["name"]
    model_config_path = resolved["release"] / release["model_config"]["name"]
    if _sha256(training_path) != release["training_config"]["sha256"]:
        raise PublicBaseEvaluationError("release training config hash mismatch")
    if _sha256(model_config_path) != release["model_config"]["sha256"]:
        raise PublicBaseEvaluationError("release model config hash mismatch")
    if release.get("checks", {}).get("validation_deferred") is not True or release.get(
        "validation"
    ) != {"status": "deferred", "dataset": None}:
        raise PublicBaseEvaluationError("release validation status is not deferred")

    completion, completion_sha = _completed(resolved["run"], "completion.json")
    if completion.get("training_config_sha256") != _sha256(training_path):
        raise PublicBaseEvaluationError(
            "completed run used a different training config"
        )
    if completion.get("final_global_step") != release["training_config"]["total_steps"]:
        raise PublicBaseEvaluationError(
            "run did not reach the released total step count"
        )
    latest_path = resolved["run"] / "checkpoints/latest.json"
    if _sha256(latest_path) != completion.get("latest_pointer_sha256"):
        raise PublicBaseEvaluationError(
            "latest checkpoint pointer changed after training"
        )
    checkpoint = resolved["run"] / "checkpoints" / completion["checkpoint_id"]
    checkpoint_manifest_path = checkpoint / "manifest.json"
    if _sha256(checkpoint_manifest_path) != completion.get(
        "checkpoint_manifest_sha256"
    ):
        raise PublicBaseEvaluationError("final checkpoint manifest hash mismatch")
    checkpoint_manifest = _json(checkpoint_manifest_path)
    if checkpoint_manifest.get("global_step") != completion["final_global_step"]:
        raise PublicBaseEvaluationError("final checkpoint step mismatch")

    tokenizer_dir, selection_sha = tokenizer_from_gpu_selection(
        resolved["selection"].relative_to(root), project_root=root
    )
    if selection_sha != release.get("tokenizer_selection_report_sha256"):
        raise PublicBaseEvaluationError("release used a different tokenizer selection")
    _, tokenizer_manifest, tokenizer_manifest_path = verify_tokenizer_directory(
        tokenizer_dir
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    suite_manifest_path = resolved["suite"] / "manifest.json"
    suite_completed_path = resolved["suite"] / "COMPLETED"
    if not suite_manifest_path.is_file() or not suite_completed_path.is_file():
        raise PublicBaseEvaluationError("benchmark suite is incomplete")
    suite_sha = _sha256(suite_manifest_path)
    if suite_completed_path.read_text(encoding="utf-8") != (
        f"manifest_sha256={suite_sha}\n"
    ):
        raise PublicBaseEvaluationError("benchmark suite marker is invalid")
    suite = _json(suite_manifest_path)
    if (
        suite.get("suite_id") != EXPECTED_SUITE_ID
        or suite.get("benchmark_counts") != EXPECTED_BENCHMARK_COUNTS
        or suite.get("task_count") != sum(EXPECTED_BENCHMARK_COUNTS.values())
        or suite.get("model_external_answering") is not False
    ):
        raise PublicBaseEvaluationError("benchmark is not the frozen full public suite")
    return {
        "run_id": completion["run_id"],
        "release_id": release["release_id"],
        "release_manifest_sha256": release_sha,
        "completion_sha256": completion_sha,
        "final_global_step": completion["final_global_step"],
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": completion["checkpoint_manifest_sha256"],
        "model_config": str(model_config_path),
        "model_config_sha256": _sha256(model_config_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "tokenizer_manifest_sha256": _sha256(tokenizer_manifest_path),
        "tokenizer_artifact_id": tokenizer_manifest.get("artifact_id"),
        "tokenizer_selection_report_sha256": selection_sha,
        "validation_status": "deferred",
        "suite": str(resolved["suite"]),
        "suite_id": suite["suite_id"],
        "suite_manifest_sha256": suite_sha,
        "benchmark_counts": suite["benchmark_counts"],
        "task_count": suite["task_count"],
        "expected_world_size": EXPECTED_WORLD_SIZE,
        "model_external_answering": False,
    }


def quality_gate(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["benchmark_metrics"]
    aggregate_chance = (
        sum(item["chance_accuracy"] * item["task_count"] for item in metrics.values())
        / report["task_count"]
    )
    thresholds = {
        "aggregate_normalized_accuracy": aggregate_chance + 0.01,
        "arc_challenge": metrics["arc_challenge"]["chance_accuracy"],
        "hellaswag": metrics["hellaswag"]["chance_accuracy"] + 0.02,
        "mmlu": metrics["mmlu"]["chance_accuracy"] - 0.01,
        "ceval": metrics["ceval"]["chance_accuracy"] - 0.01,
    }
    checks = {
        "six_gpu_complete_coverage": report.get("world_size") == EXPECTED_WORLD_SIZE
        and report.get("task_count") == sum(EXPECTED_BENCHMARK_COUNTS.values()),
        "no_model_external_answering": report.get("model_external_answering") is False,
        "aggregate_above_chance": report["normalized_accuracy"]
        >= thresholds["aggregate_normalized_accuracy"],
        **{
            f"{name}_minimum": metrics[name]["normalized_accuracy"] >= threshold
            for name, threshold in thresholds.items()
            if name in EXPECTED_BENCHMARK_COUNTS
        },
    }
    return {
        "gate_version": "public-base-delivery-gate-v2",
        "threshold_basis": (
            "statistically above-chance commissioning floor on the exact suite"
        ),
        "aggregate_chance_accuracy": aggregate_chance,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_result(output_dir: Path, report: dict[str, Any]) -> None:
    if output_dir.exists():
        raise PublicBaseEvaluationError(f"evaluation output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker = f"{_sha256(report_path)}  report.json\n"
        (temporary / "EVALUATED").write_text(marker, encoding="utf-8")
        if report["gate"]["passed"]:
            (temporary / "COMPLETED").write_text(marker, encoding="utf-8")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace, distributed: DistributedContext) -> int:
    if distributed.world_size != EXPECTED_WORLD_SIZE:
        raise PublicBaseEvaluationError("final evaluation requires exactly six GPUs")
    contract: dict[str, Any] | None = None
    if distributed.is_main_process:
        try:
            contract = {
                "ok": True,
                "value": resolve_contract(
                    run_dir=args.run_dir,
                    release_dir=args.release_dir,
                    tokenizer_selection_dir=args.tokenizer_selection_dir,
                    suite_dir=args.suite_dir,
                    project_root=args.project_root,
                ),
            }
        except BaseException as error:
            contract = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    contract = distributed.broadcast_object(contract)
    if not contract["ok"]:
        raise PublicBaseEvaluationError(
            f"rank-0 delivery verification failed: {contract['error']}"
        )
    value = contract["value"]
    benchmark = evaluate(
        Path(value["checkpoint"]),
        Path(value["suite"]),
        Path(value["model_config"]),
        Path(value["tokenizer"]),
        distributed=distributed,
    )
    if not distributed.is_main_process:
        return 0
    assert benchmark is not None
    if benchmark["suite_manifest_sha256"] != value["suite_manifest_sha256"]:
        raise PublicBaseEvaluationError("evaluated suite changed after preflight")
    report = {
        "schema_version": 1,
        "evaluation_id": "atom-base-300m-public-100b-full-v4",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": value,
        "validation": {"status": "deferred", "metrics": None},
        "benchmark": benchmark,
        "gate": quality_gate(benchmark),
        "model_external_answering": False,
    }
    _write_result(args.output_dir, report)
    print(
        json.dumps(
            {
                "evaluation_id": report["evaluation_id"],
                "gate": report["gate"],
                "benchmark_metrics": benchmark["benchmark_metrics"],
                "validation": report["validation"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["gate"]["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-selection-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    distributed = DistributedContext.initialize(
        DistributedConfig(enabled=True, backend="nccl")
    )
    try:
        return run(args, distributed)
    finally:
        distributed.close()


if __name__ == "__main__":
    raise SystemExit(main())
