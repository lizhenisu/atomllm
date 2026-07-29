"""Freeze reproducible evidence for the stage-8B training-chain gate."""

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


class SFTValidationError(RuntimeError):
    """Raised when stage-8B evidence is missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SFTValidationError(f"JSON root must be a mapping: {path}")
    return value


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _verify_latest(run: Path) -> tuple[str, str]:
    latest = _json(run / "checkpoints" / "latest.json")
    checkpoint = run / "checkpoints" / latest["checkpoint_id"]
    manifest_path = checkpoint / "manifest.json"
    if not (checkpoint / "COMPLETE").is_file():
        raise SFTValidationError(f"checkpoint is incomplete: {checkpoint}")
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != latest["manifest_sha256"]:
        raise SFTValidationError("latest checkpoint manifest hash mismatch")
    manifest = _json(manifest_path)
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise SFTValidationError(f"checkpoint payload mismatch: {name}")
    return latest["checkpoint_id"], manifest_sha


def validate(overfit_run: Path, ddp_run: Path, output: Path) -> dict[str, Any]:
    overfit = _json(overfit_run / "reports" / "training-report.json")
    ddp = _json(ddp_run / "reports" / "training-report.json")
    resume_events = _json_lines(ddp_run / "reports" / "resume-events.jsonl")
    if (
        overfit.get("completed_steps") != 20
        or overfit.get("initial_loss", 0) <= overfit.get("final_loss", 0)
        or overfit.get("formal_completion_reached") is not False
    ):
        raise SFTValidationError("single-GPU overfit evidence failed")
    if (
        ddp.get("completed_steps") != 30
        or not resume_events
        or resume_events[0].get("restored_from_step") != 10
        or resume_events[-1].get("target_step") != 30
        or ddp.get("restored_from_step") != resume_events[-1].get("restored_from_step")
        or ddp.get("formal_completion_reached") is not False
        or ddp.get("release_plan", {}).get("base_model_sha256")
        != "6017a1d5a3e95a13be9c9ad38f5bc51b9528981ea10b555f873779fdfdb662c7"
    ):
        raise SFTValidationError("6-GPU resume smoke evidence failed")
    overfit_checkpoint, overfit_sha = _verify_latest(overfit_run)
    ddp_checkpoint, ddp_sha = _verify_latest(ddp_run)
    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "single_gpu_overfit": {
            "run": str(overfit_run),
            "steps": 20,
            "initial_loss": overfit["initial_loss"],
            "final_loss": overfit["final_loss"],
            "checkpoint_id": overfit_checkpoint,
            "checkpoint_manifest_sha256": overfit_sha,
        },
        "six_gpu_resume_smoke": {
            "run": str(ddp_run),
            "steps": 30,
            "resume_chain": resume_events,
            "initial_loss": ddp["initial_loss"],
            "final_loss": ddp["final_loss"],
            "checkpoint_id": ddp_checkpoint,
            "checkpoint_manifest_sha256": ddp_sha,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_sha = _sha256(temporary / "report.json")
        (temporary / "COMPLETED").write_text(
            f"report_sha256={report_sha}\n", encoding="utf-8"
        )
        if output.exists():
            raise SFTValidationError(f"validation output already exists: {output}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and freeze stage-8B evidence.")
    parser.add_argument("--overfit-run", type=Path, required=True)
    parser.add_argument("--ddp-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate(args.overfit_run, args.ddp_run, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
