"""Overfit one formal-data batch to validate the Atom-50M training connection."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager

import torch
import yaml

from atomllm.experiment import set_seed
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import file_sha256
from atomllm.training.data import ShardedTokenDataset


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/training/atom-50m-overfit.yaml")


class FixedBatchOverfitError(RuntimeError):
    """Raised when fixed-batch overfitting violates its acceptance contract."""


@dataclass(frozen=True, slots=True)
class FixedBatchOverfitConfig:
    name: str
    model_config: Path
    model_config_sha256: str
    training_data: Path
    sequence_length: int
    batch_size: int
    steps: int
    learning_rate: float
    max_gradient_norm: float
    seed: int
    device: str
    precision: str
    maximum_final_to_initial_loss_ratio: float
    output_dir: Path


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FixedBatchOverfitError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FixedBatchOverfitError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise FixedBatchOverfitError(f"{field} must be a positive integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise FixedBatchOverfitError(f"{field} must be a finite number")
    result = float(value)
    if result <= 0:
        raise FixedBatchOverfitError(f"{field} must be positive")
    return result


def load_fixed_batch_overfit_config(
    path: str | Path = DEFAULT_CONFIG,
) -> FixedBatchOverfitConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FixedBatchOverfitError(f"cannot read config: {config_path}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FixedBatchOverfitError("config must be a mapping with string keys")
    expected = {
        "schema_version",
        "name",
        "model_config",
        "model_config_sha256",
        "training_data",
        "sequence_length",
        "batch_size",
        "steps",
        "learning_rate",
        "max_gradient_norm",
        "seed",
        "device",
        "precision",
        "maximum_final_to_initial_loss_ratio",
        "output_dir",
    }
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise FixedBatchOverfitError(f"config missing fields: {', '.join(missing)}")
    if unknown:
        raise FixedBatchOverfitError(f"config has unknown fields: {', '.join(unknown)}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise FixedBatchOverfitError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise FixedBatchOverfitError("name must be a non-empty string")
    sha = value["model_config_sha256"]
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise FixedBatchOverfitError(
            "model_config_sha256 must be 64 lowercase hex digits"
        )
    seed = value["seed"]
    if type(seed) is not int or seed < 0:
        raise FixedBatchOverfitError("seed must be a non-negative integer")
    if value["device"] not in {"cpu", "cuda"}:
        raise FixedBatchOverfitError("device must be cpu or cuda")
    if value["precision"] not in {"fp32", "bf16"}:
        raise FixedBatchOverfitError("precision must be fp32 or bf16")
    if value["device"] == "cpu" and value["precision"] != "fp32":
        raise FixedBatchOverfitError("CPU overfit must use fp32")
    ratio = _positive_float(
        value["maximum_final_to_initial_loss_ratio"],
        "maximum_final_to_initial_loss_ratio",
    )
    if ratio >= 1:
        raise FixedBatchOverfitError(
            "maximum_final_to_initial_loss_ratio must be less than 1"
        )
    return FixedBatchOverfitConfig(
        name=value["name"],
        model_config=_safe_path(value["model_config"], "model_config"),
        model_config_sha256=sha,
        training_data=_safe_path(value["training_data"], "training_data"),
        sequence_length=_positive_int(value["sequence_length"], "sequence_length"),
        batch_size=_positive_int(value["batch_size"], "batch_size"),
        steps=_positive_int(value["steps"], "steps"),
        learning_rate=_positive_float(value["learning_rate"], "learning_rate"),
        max_gradient_norm=_positive_float(
            value["max_gradient_norm"], "max_gradient_norm"
        ),
        seed=seed,
        device=value["device"],
        precision=value["precision"],
        maximum_final_to_initial_loss_ratio=ratio,
        output_dir=_safe_path(value["output_dir"], "output_dir"),
    )


def _autocast(config: FixedBatchOverfitConfig) -> ContextManager[None]:
    if config.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def run_fixed_batch_overfit(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_fixed_batch_overfit_config(root / config_path)
    model_path = root / config.model_config
    if file_sha256(model_path) != config.model_config_sha256:
        raise FixedBatchOverfitError("model config SHA-256 mismatch")
    model_config = load_model_config(model_path)
    if config.sequence_length > model_config.dimensions.max_sequence_length:
        raise FixedBatchOverfitError("sequence length exceeds model context")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise FixedBatchOverfitError("CUDA is unavailable")
    dataset = ShardedTokenDataset(
        root / config.training_data,
        sequence_length=config.sequence_length,
    )
    if config.batch_size > len(dataset):
        raise FixedBatchOverfitError("batch size exceeds dataset")
    batch = torch.stack([dataset[index] for index in range(config.batch_size)])
    set_seed(config.seed)
    device = torch.device(config.device)
    model = AtomLLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    batch = batch.to(device)
    losses: list[float] = []
    gradient_norms: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config):
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss).item():
            raise FixedBatchOverfitError(f"non-finite loss at step {step}")
        output.loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
        )
        if not math.isfinite(gradient_norm):
            raise FixedBatchOverfitError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        losses.append(float(output.loss.detach()))
        gradient_norms.append(gradient_norm)
        if step == 1 or step % 10 == 0 or step == config.steps:
            print(
                f"[fixed-batch-overfit] step={step}/{config.steps} "
                f"loss={losses[-1]:.6f} grad_norm={gradient_norm:.4f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ratio = losses[-1] / losses[0]
    passed = ratio <= config.maximum_final_to_initial_loss_ratio
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "model": model_config.name,
        "parameter_count": model_config.expected_parameter_count,
        "dataset_id": dataset.dataset_id,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "sequence_length": config.sequence_length,
        "batch_size": config.batch_size,
        "steps": config.steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_to_initial_loss_ratio": ratio,
        "maximum_allowed_ratio": config.maximum_final_to_initial_loss_ratio,
        "minimum_loss": min(losses),
        "maximum_gradient_norm": max(gradient_norms),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "passed": passed,
    }
    _write_report_atomic(root / config.output_dir / "report.json", report)
    if not passed:
        raise FixedBatchOverfitError(
            f"final/initial loss ratio {ratio:.4f} exceeds "
            f"{config.maximum_final_to_initial_loss_ratio:.4f}"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fixed_batch_overfit(args.config, project_root=args.project_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
