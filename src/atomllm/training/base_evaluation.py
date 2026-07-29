"""Distributed held-out loss evaluation for frozen AtomLLM checkpoints."""

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

import torch
import torch.distributed as dist
import yaml

from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.post_training.sft_training import _verify_base_checkpoint
from atomllm.training.config import DistributedConfig
from atomllm.training.data import ShardedTokenDataset
from atomllm.training.distributed import DistributedContext


class BaseEvaluationError(RuntimeError):
    """Raised when a frozen base evaluation contract is violated."""


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
        "validation_data",
        "validation_manifest_sha256",
        "sequence_length",
        "sample_blocks",
        "sample_seed",
        "micro_batch_size",
        "loss_chunk_size",
        "expected_world_size",
        "checkpoints",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise BaseEvaluationError(
            f"evaluation config fields must be {sorted(required)}"
        )
    if config["schema_version"] != 1:
        raise BaseEvaluationError("unsupported evaluation schema")
    for name in (
        "sequence_length",
        "sample_blocks",
        "micro_batch_size",
        "loss_chunk_size",
        "expected_world_size",
    ):
        if type(config[name]) is not int or config[name] <= 0:
            raise BaseEvaluationError(f"{name} must be a positive integer")
    if type(config["sample_seed"]) is not int or config["sample_seed"] < 0:
        raise BaseEvaluationError("sample_seed must be a non-negative integer")
    if not isinstance(config["checkpoints"], list) or not config["checkpoints"]:
        raise BaseEvaluationError("checkpoints must be a non-empty list")
    checkpoint_fields = {
        "name",
        "path",
        "manifest_sha256",
        "model_sha256",
    }
    names: set[str] = set()
    for checkpoint in config["checkpoints"]:
        if not isinstance(checkpoint, dict) or set(checkpoint) != checkpoint_fields:
            raise BaseEvaluationError(
                f"checkpoint fields must be {sorted(checkpoint_fields)}"
            )
        if checkpoint["name"] in names:
            raise BaseEvaluationError("checkpoint names must be unique")
        names.add(checkpoint["name"])
    return config


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _broadcast_dataset(
    directory: Path, sequence_length: int, distributed: DistributedContext
) -> ShardedTokenDataset:
    payload: dict[str, Any] | None = None
    primary: ShardedTokenDataset | None = None
    if distributed.is_main_process:
        try:
            primary = ShardedTokenDataset(directory, sequence_length=sequence_length)
            payload = {
                "ok": True,
                "manifest": primary.manifest,
                "manifest_sha256": primary.manifest_sha256,
            }
        except BaseException as error:
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = distributed.broadcast_object(payload)
    if not payload["ok"]:
        raise BaseEvaluationError(
            f"rank-0 validation verification failed: {payload['error']}"
        )
    return primary or ShardedTokenDataset(
        directory,
        sequence_length=sequence_length,
        verified_manifest=payload["manifest"],
        manifest_sha256=payload["manifest_sha256"],
    )


def _sample_indices(
    block_count: int,
    sample_blocks: int,
    seed: int,
    distributed: DistributedContext,
) -> list[int]:
    if sample_blocks > block_count:
        raise BaseEvaluationError("sample_blocks exceeds validation block count")
    indices: list[int] | None = None
    if distributed.is_main_process:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(block_count, generator=generator)[
            :sample_blocks
        ].tolist()
    return distributed.broadcast_object(indices)


@torch.inference_mode()
def _evaluate_checkpoint(
    model: AtomLLM,
    dataset: ShardedTokenDataset,
    indices: list[int],
    *,
    micro_batch_size: int,
    loss_chunk_size: int,
    distributed: DistributedContext,
) -> dict[str, Any]:
    device = distributed.device("cuda")
    local_indices = indices[distributed.rank :: distributed.world_size]
    local_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    local_targets = torch.zeros((), device=device, dtype=torch.int64)
    distributed.barrier()
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(local_indices), micro_batch_size):
        batch_indices = local_indices[start : start + micro_batch_size]
        batch = torch.stack([dataset[index] for index in batch_indices]).to(device)
        target_count = (batch[:, 1:] != model.pad_token_id).sum()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch, loss_chunk_size=loss_chunk_size)
        if output.loss is None or not torch.isfinite(output.loss).item():
            raise BaseEvaluationError("validation loss is not finite")
        local_loss_sum += output.loss.double() * target_count
        local_targets += target_count
    elapsed = torch.tensor(
        time.perf_counter() - started, device=device, dtype=torch.float64
    )
    if distributed.is_distributed:
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_targets, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    loss = float(local_loss_sum / local_targets)
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "evaluated_target_tokens": int(local_targets),
        "elapsed_seconds": float(elapsed),
        "target_tokens_per_second": float(local_targets / elapsed),
    }


def run(args: argparse.Namespace, distributed: DistributedContext) -> int:
    config = _load_config(args.config)
    if distributed.world_size != config["expected_world_size"]:
        raise BaseEvaluationError("runtime world size does not match evaluation config")
    model_config_path = Path(config["model_config"])
    validation_path = Path(config["validation_data"])
    if _sha256(model_config_path) != config["model_config_sha256"]:
        raise BaseEvaluationError("model config SHA-256 does not match")
    dataset = _broadcast_dataset(
        validation_path, config["sequence_length"], distributed
    )
    if dataset.manifest_sha256 != config["validation_manifest_sha256"]:
        raise BaseEvaluationError("validation manifest SHA-256 does not match")
    if dataset.manifest.get("split") != "validation":
        raise BaseEvaluationError("evaluation data is not the validation split")
    indices = _sample_indices(
        len(dataset), config["sample_blocks"], config["sample_seed"], distributed
    )
    index_sha = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()
    ).hexdigest()
    model_config = load_model_config(model_config_path)
    device = distributed.device("cuda")
    model = AtomLLM(model_config).to(device=device, dtype=torch.float32)
    results: list[dict[str, Any]] = []
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
            raise BaseEvaluationError(
                f"checkpoint verification failed: {verification['error']}"
            )
        distributed.barrier()
        load_safetensors_checkpoint(model, directory / "model.safetensors")
        metrics = _evaluate_checkpoint(
            model,
            dataset,
            indices,
            micro_batch_size=config["micro_batch_size"],
            loss_chunk_size=config["loss_chunk_size"],
            distributed=distributed,
        )
        result = {
            "name": checkpoint["name"],
            "checkpoint": str(directory),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "model_sha256": checkpoint["model_sha256"],
            **metrics,
        }
        results.append(result)
        if distributed.is_main_process:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if distributed.is_main_process:
        report = {
            "schema_version": 1,
            "evaluation_id": config["evaluation_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "config_sha256": _sha256(args.config),
            "model_config_sha256": config["model_config_sha256"],
            "validation_manifest_sha256": dataset.manifest_sha256,
            "validation_dataset_id": dataset.dataset_id,
            "sequence_length": config["sequence_length"],
            "sample_blocks": len(indices),
            "sample_seed": config["sample_seed"],
            "sample_indices_sha256": index_sha,
            "world_size": distributed.world_size,
            "results": results,
        }
        _atomic_json(args.output, report)
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
