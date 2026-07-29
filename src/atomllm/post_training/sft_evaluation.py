"""Evaluate assistant-only loss on a frozen held-out SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.post_training.sft_data import INPUT_FILE, LABEL_FILE
from atomllm.post_training.sft_heldout_data import verify_dataset
from atomllm.post_training.sft_training import _verify_base_checkpoint
from atomllm.training.config import DistributedConfig
from atomllm.training.distributed import DistributedContext


class SFTEvaluationError(RuntimeError):
    """Raised when held-out SFT evaluation violates its frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "evaluation_id",
        "model_config",
        "model_config_sha256",
        "heldout_data",
        "heldout_manifest_sha256",
        "micro_batch_size",
        "loss_chunk_size",
        "expected_world_size",
        "checkpoints",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise SFTEvaluationError(
            f"SFT evaluation config fields must be {sorted(required)}"
        )
    if config["schema_version"] != 1:
        raise SFTEvaluationError("unsupported SFT evaluation schema")
    for name in ("micro_batch_size", "loss_chunk_size", "expected_world_size"):
        if type(config[name]) is not int or config[name] <= 0:
            raise SFTEvaluationError(f"{name} must be a positive integer")
    fields = {"name", "path", "manifest_sha256", "model_sha256"}
    if not isinstance(config["checkpoints"], list) or not config["checkpoints"]:
        raise SFTEvaluationError("checkpoints must be a non-empty list")
    names: set[str] = set()
    for checkpoint in config["checkpoints"]:
        if not isinstance(checkpoint, dict) or set(checkpoint) != fields:
            raise SFTEvaluationError(f"checkpoint fields must be {sorted(fields)}")
        if checkpoint["name"] in names:
            raise SFTEvaluationError("checkpoint names must be unique")
        names.add(checkpoint["name"])
    return config


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _broadcast_manifest(
    directory: Path, expected_sha: str, distributed: DistributedContext
) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    if distributed.is_main_process:
        try:
            manifest = verify_dataset(directory)
            actual_sha = _sha256(directory / "manifest.json")
            if actual_sha != expected_sha:
                raise SFTEvaluationError("held-out manifest SHA-256 does not match")
            payload = {"ok": True, "manifest": manifest}
        except BaseException as error:
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = distributed.broadcast_object(payload)
    if not payload["ok"]:
        raise SFTEvaluationError(
            f"rank-0 held-out verification failed: {payload['error']}"
        )
    return payload["manifest"]


@torch.inference_mode()
def _evaluate(
    model: AtomLLM,
    inputs: np.memmap,
    labels: np.memmap,
    block_count: int,
    *,
    micro_batch_size: int,
    loss_chunk_size: int,
    distributed: DistributedContext,
) -> dict[str, Any]:
    device = distributed.device("cuda")
    local_indices = list(range(distributed.rank, block_count, distributed.world_size))
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    targets = torch.zeros((), device=device, dtype=torch.int64)
    distributed.barrier()
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(local_indices), micro_batch_size):
        indices = local_indices[start : start + micro_batch_size]
        batch_inputs = torch.from_numpy(np.array(inputs[indices], dtype=np.int64)).to(
            device
        )
        batch_labels = torch.from_numpy(np.array(labels[indices], dtype=np.int64)).to(
            device
        )
        target_count = (batch_labels != -100).sum()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                batch_inputs,
                labels=batch_labels,
                loss_chunk_size=loss_chunk_size,
            )
        if output.loss is None or not torch.isfinite(output.loss):
            raise SFTEvaluationError("held-out SFT loss is not finite")
        loss_sum += output.loss.double() * target_count
        targets += target_count
    elapsed = torch.tensor(
        time.perf_counter() - started, device=device, dtype=torch.float64
    )
    if distributed.is_distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(targets, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    loss = float(loss_sum / targets)
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "evaluated_assistant_target_tokens": int(targets),
        "elapsed_seconds": float(elapsed),
        "assistant_target_tokens_per_second": float(targets / elapsed),
    }


def run(args: argparse.Namespace, distributed: DistributedContext) -> int:
    config = _load_config(args.config)
    if distributed.world_size != config["expected_world_size"]:
        raise SFTEvaluationError("runtime world size does not match config")
    model_config_path = Path(config["model_config"])
    if _sha256(model_config_path) != config["model_config_sha256"]:
        raise SFTEvaluationError("model config SHA-256 does not match")
    data_path = Path(config["heldout_data"])
    manifest = _broadcast_manifest(
        data_path, config["heldout_manifest_sha256"], distributed
    )
    shape = (manifest["block_count"], manifest["sequence_length"])
    inputs = np.memmap(data_path / INPUT_FILE, mode="r", dtype="<u4", shape=shape)
    labels = np.memmap(data_path / LABEL_FILE, mode="r", dtype="<i4", shape=shape)
    model = AtomLLM(load_model_config(model_config_path)).to(
        device=distributed.device("cuda"), dtype=torch.float32
    )
    results = []
    for checkpoint in config["checkpoints"]:
        verification: dict[str, Any] | None = None
        directory = Path(checkpoint["path"])
        if distributed.is_main_process:
            try:
                _verify_base_checkpoint(
                    directory,
                    expected_manifest_sha256=checkpoint["manifest_sha256"],
                    expected_model_sha256=checkpoint["model_sha256"],
                )
                verification = {"ok": True}
            except BaseException as error:
                verification = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        verification = distributed.broadcast_object(verification)
        if not verification["ok"]:
            raise SFTEvaluationError(
                f"checkpoint verification failed: {verification['error']}"
            )
        load_safetensors_checkpoint(model, directory / "model.safetensors")
        metrics = _evaluate(
            model,
            inputs,
            labels,
            manifest["block_count"],
            micro_batch_size=config["micro_batch_size"],
            loss_chunk_size=config["loss_chunk_size"],
            distributed=distributed,
        )
        result = {"name": checkpoint["name"], **checkpoint, **metrics}
        results.append(result)
        if distributed.is_main_process:
            print(json.dumps(result, sort_keys=True), flush=True)
    if distributed.is_main_process:
        _atomic_json(
            args.output,
            {
                "schema_version": 1,
                "evaluation_id": config["evaluation_id"],
                "created_at": datetime.now(UTC).isoformat(),
                "config_sha256": _sha256(args.config),
                "heldout_manifest_sha256": config["heldout_manifest_sha256"],
                "heldout_dataset_id": manifest["dataset_id"],
                "heldout_examples": manifest["eligible_examples"],
                "world_size": distributed.world_size,
                "results": results,
            },
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preview = _load_config(args.config)
    distributed = DistributedContext.initialize(
        DistributedConfig(enabled=preview["expected_world_size"] > 1, backend="nccl")
    )
    try:
        return run(args, distributed)
    finally:
        distributed.close()


if __name__ == "__main__":
    raise SystemExit(main())
