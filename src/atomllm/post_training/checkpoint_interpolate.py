"""Create auditable linear interpolations between compatible AtomLLM weights."""

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

import torch

from atomllm.model.checkpoint import (
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM


class CheckpointInterpolationError(RuntimeError):
    """Raised when source weights cannot be safely interpolated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    model_path = directory / "model.safetensors"
    if not (directory / "COMPLETE").is_file() or not manifest_path.is_file():
        raise CheckpointInterpolationError(f"checkpoint is incomplete: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("files", {}).get("model.safetensors")
    if (
        not isinstance(record, dict)
        or not model_path.is_file()
        or model_path.stat().st_size != record.get("bytes")
        or _sha256(model_path) != record.get("sha256")
    ):
        raise CheckpointInterpolationError(f"model payload mismatch: {directory}")
    return {
        "path": str(directory),
        "manifest_sha256": _sha256(manifest_path),
        "model_sha256": record["sha256"],
    }


@torch.no_grad()
def _advance_interpolation(
    current: torch.nn.Module,
    target: torch.nn.Module,
    *,
    current_alpha: float,
    target_alpha: float,
) -> None:
    if not 0.0 <= current_alpha < target_alpha <= 1.0:
        raise CheckpointInterpolationError("alphas must increase within (0, 1]")
    current_parameters = dict(current.named_parameters())
    target_parameters = dict(target.named_parameters())
    if current_parameters.keys() != target_parameters.keys():
        raise CheckpointInterpolationError("model parameter names do not match")
    relative_alpha = (target_alpha - current_alpha) / (1.0 - current_alpha)
    for name, parameter in current_parameters.items():
        target_parameter = target_parameters[name]
        if parameter.shape != target_parameter.shape:
            raise CheckpointInterpolationError(f"parameter shape mismatch: {name}")
        parameter.lerp_(target_parameter, relative_alpha)


def interpolate(
    model_config: Path,
    base_checkpoint: Path,
    target_checkpoint: Path,
    alphas: list[float],
    output_run_dir: Path,
) -> list[dict[str, Any]]:
    if not alphas or alphas != sorted(set(alphas)):
        raise CheckpointInterpolationError("alphas must be unique and increasing")
    if any(not 0.0 < alpha < 1.0 for alpha in alphas):
        raise CheckpointInterpolationError("interpolation alphas must be in (0, 1)")
    base = _verify_checkpoint(base_checkpoint)
    target = _verify_checkpoint(target_checkpoint)
    config = load_model_config(model_config)
    current_model = AtomLLM(config).to(dtype=torch.float32)
    target_model = AtomLLM(config).to(dtype=torch.float32)
    load_safetensors_checkpoint(current_model, base_checkpoint / "model.safetensors")
    load_safetensors_checkpoint(target_model, target_checkpoint / "model.safetensors")

    checkpoints = output_run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    current_alpha = 0.0
    try:
        for alpha in alphas:
            _advance_interpolation(
                current_model,
                target_model,
                current_alpha=current_alpha,
                target_alpha=alpha,
            )
            name = f"step-{round(alpha * 1000):09d}"
            temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=checkpoints))
            model_path = temporary / "model.safetensors"
            save_safetensors_checkpoint(current_model, model_path)
            model_record = {
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
            }
            manifest = {
                "format_version": 1,
                "checkpoint_kind": "linear-weight-interpolation-v1",
                "checkpoint_id": name,
                "created_at": datetime.now(UTC).isoformat(),
                "alpha": alpha,
                "model_config_sha256": _sha256(model_config),
                "base_checkpoint": base,
                "target_checkpoint": target,
                "files": {"model.safetensors": model_record},
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (temporary / "COMPLETE").write_text(
                "atomllm-sft-checkpoint-v1\n", encoding="utf-8"
            )
            destination = checkpoints / name
            os.replace(temporary, destination)
            result = {
                "checkpoint": str(destination),
                "alpha": alpha,
                "manifest_sha256": _sha256(destination / "manifest.json"),
                "model_sha256": model_record["sha256"],
            }
            results.append(result)
            current_alpha = alpha
        latest = results[-1]
        (checkpoints / "latest.json").write_text(
            json.dumps(
                {
                    "checkpoint_id": Path(latest["checkpoint"]).name,
                    "manifest_sha256": latest["manifest_sha256"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_run_dir / "interpolation-report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "base_checkpoint": base,
                    "target_checkpoint": target,
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return results
    except BaseException:
        shutil.rmtree(output_run_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    results = interpolate(
        args.model_config,
        args.base_checkpoint,
        args.target_checkpoint,
        args.alphas,
        args.output_run_dir,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
