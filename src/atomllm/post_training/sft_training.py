"""DDP SFT runtime with full-coverage release gates and exact resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel

from atomllm.model.checkpoint import (
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.post_training.sft_data import (
    BOS_ID,
    EOT_ID,
    IGNORE_INDEX,
    INPUT_FILE,
    LABEL_FILE,
    PAD_ID,
    verify_dataset,
)
from atomllm.training.config import DistributedConfig
from atomllm.training.data import ResumableShardedBatchIterator, ShardedTokenDataset
from atomllm.training.distributed import DistributedContext


class SFTTrainingError(RuntimeError):
    """Raised when SFT training violates the stage-8 release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact(data: dict[str, Any], fields: set[str], context: str) -> None:
    if set(data) != fields:
        raise SFTTrainingError(f"{context} fields must be {sorted(fields)}")


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SFTTrainingError(f"{name} must be a positive integer")
    return value


def _verify_base_checkpoint(
    directory: Path,
    *,
    expected_manifest_sha256: str | None,
    expected_model_sha256: str | None,
) -> None:
    """Verify either a pretraining-v2 or SFT-v1 initialization checkpoint."""
    manifest_path = directory / "manifest.json"
    complete_path = directory / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise SFTTrainingError("base checkpoint is incomplete")
    if complete_path.read_text(encoding="utf-8") not in {
        "atomllm-checkpoint-complete-v1\n",
        "atomllm-sft-checkpoint-v1\n",
    }:
        raise SFTTrainingError("base checkpoint COMPLETE marker is unsupported")
    manifest_sha = _sha256(manifest_path)
    if expected_manifest_sha256 is None or manifest_sha != expected_manifest_sha256:
        raise SFTTrainingError("base checkpoint manifest SHA-256 does not match")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("files", {}).get("model.safetensors")
    if not isinstance(record, dict):
        raise SFTTrainingError("base checkpoint model record is missing")
    model_path = directory / "model.safetensors"
    expected_size = record.get("bytes", record.get("size_bytes"))
    if (
        not model_path.is_file()
        or model_path.stat().st_size != expected_size
        or _sha256(model_path) != record.get("sha256")
        or expected_model_sha256 is None
        or record.get("sha256") != expected_model_sha256
    ):
        raise SFTTrainingError("base checkpoint model payload does not match")


@dataclass(frozen=True, slots=True)
class SFTConfig:
    path: Path
    name: str
    status: str
    seed: int
    model_config: Path
    expected_parameters: int
    dataset: Path
    dataset_manifest_sha256: str
    required_data_policy: str
    base_checkpoint: Path | None
    base_manifest_sha256: str | None
    base_model_sha256: str | None
    base_validation_status: str
    world_size: int
    micro_batch_size: int
    accumulation_steps: int
    learning_rate: float
    minimum_learning_rate_ratio: float
    warmup_ratio: float
    weight_decay: float
    max_gradient_norm: float
    checkpoint_every: int
    keep_last: int
    precision: str
    gradient_checkpointing: bool
    checkpoint_segment_layers: int
    checkpoint_interval_segments: int
    loss_chunk_size: int
    ddp_bucket_cap_mb: int
    isolate_packed_conversations: bool
    eot_loss_weight: float
    lower_target_tokens: int
    upper_target_tokens: int
    replay_dataset: Path | None
    replay_manifest_sha256: str | None
    replay_interval_steps: int | None
    replay_loss_weight: float
    replay_seed: int | None


def load_sft_config(path: Path, project_root: Path = Path(".")) -> SFTConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SFTTrainingError("SFT config must be a mapping")
    schema_version = raw.get("schema_version")
    top_level_fields = {
        "schema_version",
        "name",
        "status",
        "seed",
        "model",
        "data",
        "initialization",
        "budget",
        "batch",
        "optimizer",
        "checkpoint",
        "runtime",
    }
    if schema_version == 2:
        top_level_fields.add("replay")
    _exact(raw, top_level_fields, "SFT config")
    if schema_version not in {1, 2} or raw["status"] not in {"smoke", "release"}:
        raise SFTTrainingError("unsupported SFT schema or status")
    model = raw["model"]
    data = raw["data"]
    init = raw["initialization"]
    budget = raw["budget"]
    batch = raw["batch"]
    optimizer = raw["optimizer"]
    checkpoint = raw["checkpoint"]
    runtime = raw["runtime"]
    replay = raw.get("replay")
    _exact(model, {"config_path", "expected_parameter_count"}, "model")
    data_fields = {
        "path",
        "manifest_sha256",
        "sample_ratio",
        "max_samples",
        "drop_last",
    }
    if "required_policy" in data:
        data_fields.add("required_policy")
    _exact(data, data_fields, "data")
    _exact(
        init,
        {
            "base_checkpoint",
            "base_manifest_sha256",
            "base_model_sha256",
            "load_optimizer_state",
            "base_validation_status",
        },
        "initialization",
    )
    _exact(
        budget,
        {"lower_assistant_target_tokens", "upper_assistant_target_tokens"},
        "budget",
    )
    _exact(
        batch,
        {"expected_world_size", "micro_batch_size", "gradient_accumulation_steps"},
        "batch",
    )
    _exact(
        optimizer,
        {
            "learning_rate",
            "minimum_learning_rate_ratio",
            "warmup_ratio",
            "weight_decay",
            "max_gradient_norm",
        },
        "optimizer",
    )
    _exact(checkpoint, {"save_every_steps", "keep_last", "exact_resume"}, "checkpoint")
    runtime_fields = {
        "device",
        "precision",
        "gradient_checkpointing",
        "checkpoint_segment_layers",
        "checkpoint_interval_segments",
        "loss_chunk_size",
        "ddp_bucket_cap_mb",
        "compile_model",
    }
    if "isolate_packed_conversations" in runtime:
        runtime_fields.add("isolate_packed_conversations")
    if "eot_loss_weight" in runtime:
        runtime_fields.add("eot_loss_weight")
    _exact(runtime, runtime_fields, "runtime")
    if schema_version == 2:
        if not isinstance(replay, dict):
            raise SFTTrainingError("schema v2 requires replay configuration")
        _exact(
            replay,
            {"path", "manifest_sha256", "interval_steps", "loss_weight", "seed"},
            "replay",
        )
    if (
        data["sample_ratio"] != 1
        or data["max_samples"] is not None
        or data["drop_last"] is not False
    ):
        raise SFTTrainingError(
            "SFT data must use sample_ratio=1, max_samples=null, and drop_last=false"
        )
    required_data_policy = data.get("required_policy", "any")
    if required_data_policy not in {"any", "public-only"}:
        raise SFTTrainingError("unsupported required SFT data policy")
    if init["load_optimizer_state"] is not False:
        raise SFTTrainingError("SFT must initialize a fresh optimizer")
    validation_status = init["base_validation_status"]
    allowed_validation_statuses = (
        {"deferred"}
        if schema_version == 1
        else {
            "deferred",
            "passed",
        }
    )
    if validation_status not in allowed_validation_statuses:
        raise SFTTrainingError("base validation status is unsupported")
    if (
        schema_version == 2
        and raw["status"] == "release"
        and validation_status != "passed"
    ):
        raise SFTTrainingError("schema v2 release SFT requires a validated base")
    if checkpoint["exact_resume"] is not True or runtime["compile_model"] is not False:
        raise SFTTrainingError(
            "exact resume is required and compile_model is unsupported"
        )
    if runtime["device"] != "cuda" or runtime["precision"] != "bf16":
        raise SFTTrainingError("stage-8 SFT requires CUDA BF16")
    isolate_packed_conversations = runtime.get("isolate_packed_conversations", False)
    if type(isolate_packed_conversations) is not bool:
        raise SFTTrainingError("runtime.isolate_packed_conversations must be boolean")
    eot_loss_weight = runtime.get("eot_loss_weight", 1.0)
    if type(eot_loss_weight) not in {int, float} or not 0 < float(eot_loss_weight) <= 1:
        raise SFTTrainingError("runtime.eot_loss_weight must be in (0, 1]")
    if (
        type(budget["lower_assistant_target_tokens"]) is not int
        or type(budget["upper_assistant_target_tokens"]) is not int
        or budget["lower_assistant_target_tokens"] <= 0
        or budget["lower_assistant_target_tokens"]
        > budget["upper_assistant_target_tokens"]
    ):
        raise SFTTrainingError("SFT token budget must be a positive ordered interval")
    root = project_root.resolve()
    base_value = init["base_checkpoint"]
    replay_path = None if replay is None else root / replay["path"]
    replay_manifest = None if replay is None else str(replay["manifest_sha256"])
    replay_interval = (
        None
        if replay is None
        else _positive(replay["interval_steps"], "replay.interval_steps")
    )
    replay_weight = 0.0 if replay is None else float(replay["loss_weight"])
    replay_seed = None if replay is None else _positive(replay["seed"], "replay.seed")
    if replay is not None and not 0.0 < replay_weight <= 1.0:
        raise SFTTrainingError("replay.loss_weight must be in (0, 1]")
    if replay is not None and len(replay_manifest) != 64:
        raise SFTTrainingError("replay.manifest_sha256 must be a SHA-256 digest")
    config = SFTConfig(
        path=path,
        name=str(raw["name"]),
        status=raw["status"],
        seed=int(raw["seed"]),
        model_config=root / model["config_path"],
        expected_parameters=_positive(
            model["expected_parameter_count"], "model.expected_parameter_count"
        ),
        dataset=root / data["path"],
        dataset_manifest_sha256=str(data["manifest_sha256"]),
        required_data_policy=required_data_policy,
        base_checkpoint=None if base_value is None else root / base_value,
        base_manifest_sha256=init["base_manifest_sha256"],
        base_model_sha256=init["base_model_sha256"],
        base_validation_status=validation_status,
        world_size=_positive(batch["expected_world_size"], "batch.expected_world_size"),
        micro_batch_size=_positive(batch["micro_batch_size"], "batch.micro_batch_size"),
        accumulation_steps=_positive(
            batch["gradient_accumulation_steps"], "batch.gradient_accumulation_steps"
        ),
        learning_rate=float(optimizer["learning_rate"]),
        minimum_learning_rate_ratio=float(optimizer["minimum_learning_rate_ratio"]),
        warmup_ratio=float(optimizer["warmup_ratio"]),
        weight_decay=float(optimizer["weight_decay"]),
        max_gradient_norm=float(optimizer["max_gradient_norm"]),
        checkpoint_every=_positive(
            checkpoint["save_every_steps"], "checkpoint.save_every_steps"
        ),
        keep_last=_positive(checkpoint["keep_last"], "checkpoint.keep_last"),
        precision=runtime["precision"],
        gradient_checkpointing=bool(runtime["gradient_checkpointing"]),
        checkpoint_segment_layers=_positive(
            runtime["checkpoint_segment_layers"], "runtime.checkpoint_segment_layers"
        ),
        checkpoint_interval_segments=_positive(
            runtime["checkpoint_interval_segments"],
            "runtime.checkpoint_interval_segments",
        ),
        loss_chunk_size=_positive(
            runtime["loss_chunk_size"], "runtime.loss_chunk_size"
        ),
        ddp_bucket_cap_mb=_positive(
            runtime["ddp_bucket_cap_mb"], "runtime.ddp_bucket_cap_mb"
        ),
        isolate_packed_conversations=isolate_packed_conversations,
        eot_loss_weight=float(eot_loss_weight),
        lower_target_tokens=_positive(
            budget["lower_assistant_target_tokens"],
            "budget.lower_assistant_target_tokens",
        ),
        upper_target_tokens=_positive(
            budget["upper_assistant_target_tokens"],
            "budget.upper_assistant_target_tokens",
        ),
        replay_dataset=replay_path,
        replay_manifest_sha256=replay_manifest,
        replay_interval_steps=replay_interval,
        replay_loss_weight=replay_weight,
        replay_seed=replay_seed,
    )
    if config.status == "release" and config.base_checkpoint is None:
        raise SFTTrainingError("release SFT requires a frozen base checkpoint")
    return config


def _verify_training_data_policy(
    manifest: dict[str, Any], required_policy: str
) -> None:
    if required_policy == "any":
        return
    if required_policy != "public-only":
        raise SFTTrainingError("unsupported required SFT data policy")
    if manifest.get("data_policy") != "public-only":
        raise SFTTrainingError("training requires a public-only dataset manifest")
    source_contract = manifest.get("source_contract")
    if not isinstance(source_contract, dict) or not source_contract:
        raise SFTTrainingError("public-only dataset has no source contract")
    required_source_fields = {"repository", "revision", "license", "files"}
    for name, contract in source_contract.items():
        if name.startswith("synthetic") or not isinstance(contract, dict):
            raise SFTTrainingError("public-only dataset contains a synthetic source")
        if not required_source_fields <= set(contract):
            raise SFTTrainingError(
                f"public-only source is not reproducibly pinned: {name}"
            )
    source_targets = manifest.get("source_assistant_target_tokens")
    if not isinstance(source_targets, dict) or not source_targets:
        raise SFTTrainingError("public-only dataset has no source token accounting")
    if any(name.startswith("synthetic") for name in source_targets):
        raise SFTTrainingError("public-only dataset contains synthetic target tokens")


class SFTDataset:
    def __init__(
        self,
        directory: Path,
        *,
        manifest: dict[str, Any] | None = None,
        target_counts: np.ndarray | None = None,
    ) -> None:
        self.directory = directory
        self.manifest = verify_dataset(directory) if manifest is None else manifest
        self.sequence_length = self.manifest["sequence_length"]
        self.block_count = self.manifest["block_count"]
        shape = (self.block_count, self.sequence_length)
        self.inputs = np.memmap(
            directory / INPUT_FILE, mode="r", dtype="<u4", shape=shape
        )
        self.labels = np.memmap(
            directory / LABEL_FILE, mode="r", dtype="<i4", shape=shape
        )
        if target_counts is None:
            self.target_counts = np.empty(self.block_count, dtype=np.int64)
            for start in range(0, self.block_count, 1024):
                end = min(start + 1024, self.block_count)
                self.target_counts[start:end] = np.count_nonzero(
                    self.labels[start:end] != IGNORE_INDEX, axis=1
                )
        else:
            self.target_counts = np.asarray(target_counts, dtype=np.int64)
            if self.target_counts.shape != (self.block_count,):
                raise SFTTrainingError("broadcast target counts have invalid shape")
        if int(self.target_counts.sum()) != self.manifest.get(
            "packed_assistant_target_tokens",
            self.manifest["unique_assistant_target_tokens"],
        ):
            raise SFTTrainingError(
                "assistant target count disagrees with dataset manifest"
            )
        target_indices = np.flatnonzero(self.target_counts > 0)
        if not target_indices.size:
            raise SFTTrainingError("SFT dataset has no supervised target blocks")
        self.fallback_target_index = int(target_indices[0])
        self.unique_target_tokens = self.manifest["unique_assistant_target_tokens"]
        self.packed_target_tokens = int(self.target_counts.sum())

    def batch(
        self, indices: list[int], valid: list[bool]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        safe = [
            index if flag else 0 for index, flag in zip(indices, valid, strict=True)
        ]
        inputs = torch.from_numpy(np.array(self.inputs[safe], dtype=np.int64))
        labels = torch.from_numpy(np.array(self.labels[safe], dtype=np.int64))
        valid_targets = 0
        for row, (index, flag) in enumerate(zip(indices, valid, strict=True)):
            if flag:
                valid_targets += int(self.target_counts[index])
            elif any(valid):
                labels[row].fill_(IGNORE_INDEX)
        if valid_targets == 0:
            # All DDP ranks must execute matching collectives. A long-prompt
            # fragment can legitimately contain no assistant labels. Use a
            # supervised surrogate solely to build a zero-weight gradient; the
            # caller retains local_targets=0, so this adds no loss or tokens.
            fallback = [self.fallback_target_index] * len(indices)
            inputs = torch.from_numpy(np.array(self.inputs[fallback], dtype=np.int64))
            labels = torch.from_numpy(np.array(self.labels[fallback], dtype=np.int64))
        return inputs, labels, valid_targets


def _packed_segment_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Assign each BOS-delimited packed conversation an independent segment."""
    if input_ids.ndim != 2:
        raise SFTTrainingError("packed SFT input must have shape [batch, sequence]")
    if input_ids.dtype not in {torch.int32, torch.int64}:
        raise SFTTrainingError("packed SFT input must use an integer dtype")
    non_padding = input_ids != PAD_ID
    if not torch.all((input_ids[:, 0] == BOS_ID) | ~non_padding[:, 0]).item():
        raise SFTTrainingError("each non-empty packed SFT row must start with BOS")
    segment_ids = torch.cumsum(input_ids == BOS_ID, dim=1)
    return segment_ids.masked_fill(~non_padding, 0)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    steps_per_full_pass: int
    total_steps: int
    unique_target_tokens: int
    effective_target_tokens: int
    repeated_target_tokens: int


def _block_order(block_count: int, seed: int) -> np.ndarray:
    """Return one deterministic full-coverage permutation of packed blocks."""
    if block_count <= 0:
        raise SFTTrainingError("SFT dataset must contain at least one block")
    return np.random.Generator(np.random.PCG64(seed)).permutation(block_count)


def build_training_plan(dataset: SFTDataset, config: SFTConfig) -> TrainingPlan:
    global_batch = config.world_size * config.micro_batch_size
    micro_batches = math.ceil(dataset.block_count / global_batch)
    steps_per_pass = math.ceil(micro_batches / config.accumulation_steps)
    packed = int(dataset.target_counts.sum())
    unique = int(getattr(dataset, "unique_target_tokens", packed))
    block_order = _block_order(dataset.block_count, getattr(config, "seed", 0))
    if packed > config.upper_target_tokens:
        raise SFTTrainingError(
            "one full eligible pass exceeds the configured upper budget"
        )
    if packed >= config.lower_target_tokens:
        return TrainingPlan(
            steps_per_pass, steps_per_pass, unique, packed, packed - unique
        )
    total = packed
    step = steps_per_pass
    while total < config.lower_target_tokens:
        repeated_step = 0
        slot_base = (step % steps_per_pass) * config.accumulation_steps
        for micro in range(config.accumulation_steps):
            start = (slot_base + micro) * global_batch
            if start >= dataset.block_count:
                continue
            end = min(start + global_batch, dataset.block_count)
            repeated_step += int(dataset.target_counts[block_order[start:end]].sum())
        total += repeated_step
        step += 1
        if repeated_step == 0 and step % steps_per_pass == 0:
            raise SFTTrainingError("training plan made no progress")
    if total > config.upper_target_tokens:
        raise SFTTrainingError(
            "optimizer-boundary completion exceeds the configured upper budget"
        )
    return TrainingPlan(steps_per_pass, step, unique, total, total - unique)


def _lr(config: SFTConfig, plan: TrainingPlan, completed_steps: int) -> float:
    warmup = max(1, math.ceil(plan.total_steps * config.warmup_ratio))
    if completed_steps < warmup:
        return config.learning_rate * (completed_steps + 1) / warmup
    progress = (completed_steps - warmup) / max(1, plan.total_steps - warmup)
    ratio = config.minimum_learning_rate_ratio + (
        1 - config.minimum_learning_rate_ratio
    ) * 0.5 * (1 + math.cos(math.pi * progress))
    return config.learning_rate * ratio


def _optimizer(model: AtomLLM, config: SFTConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    seen: set[int] = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=all(parameter.is_cuda for parameter in (*decay, *no_decay)),
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _save_checkpoint(
    *,
    model: AtomLLM,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    effective_tokens: int,
    replay_input_tokens: int,
    config_sha: str,
    dataset_sha: str,
    distributed: DistributedContext,
    device: torch.device,
    milestone: bool,
    keep_last: int,
) -> None:
    states = distributed.all_gather_object(
        {"rank": distributed.rank, "rng": _rng_state(device)}
    )
    if distributed.is_main_process:
        checkpoints = run_dir / "checkpoints"
        name = f"step-{step:09d}"
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=checkpoints))
        try:
            save_safetensors_checkpoint(model, temporary / "model.safetensors")
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save(states, temporary / "distributed-state.pt")
            state = {
                "global_step": step,
                "effective_assistant_target_tokens": effective_tokens,
                "replay_input_tokens": replay_input_tokens,
            }
            (temporary / "trainer-state.json").write_text(
                json.dumps(state, sort_keys=True) + "\n"
            )
            files = {}
            for path in temporary.iterdir():
                files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            manifest = {
                "format_version": 1,
                "checkpoint_id": name,
                "global_step": step,
                "effective_assistant_target_tokens": effective_tokens,
                "replay_input_tokens": replay_input_tokens,
                "config_sha256": config_sha,
                "dataset_manifest_sha256": dataset_sha,
                "world_size": distributed.world_size,
                "milestone": milestone,
                "files": files,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            (temporary / "COMPLETE").write_text("atomllm-sft-checkpoint-v1\n")
            os.replace(temporary, checkpoints / name)
            latest = {
                "checkpoint_id": name,
                "manifest_sha256": _sha256(checkpoints / name / "manifest.json"),
            }
            temp_latest = checkpoints / ".latest.json.tmp"
            temp_latest.write_text(json.dumps(latest, sort_keys=True) + "\n")
            os.replace(temp_latest, checkpoints / "latest.json")
            candidates = sorted(
                path for path in checkpoints.glob("step-*") if path.is_dir()
            )
            recent = {path.name for path in candidates[-keep_last:]}
            for path in candidates:
                record = json.loads((path / "manifest.json").read_text())
                if path.name not in recent and not record["milestone"]:
                    shutil.rmtree(path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    distributed.barrier()


def _load_resume(
    model: AtomLLM,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    config_sha: str,
    dataset_sha: str,
    distributed: DistributedContext,
    device: torch.device,
) -> tuple[int, int, int]:
    latest = json.loads((run_dir / "checkpoints" / "latest.json").read_text())
    directory = run_dir / "checkpoints" / latest["checkpoint_id"]
    manifest_path = directory / "manifest.json"
    if (
        _sha256(manifest_path) != latest["manifest_sha256"]
        or not (directory / "COMPLETE").is_file()
    ):
        raise SFTTrainingError("resume checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["config_sha256"] != config_sha
        or manifest["dataset_manifest_sha256"] != dataset_sha
        or manifest["world_size"] != distributed.world_size
    ):
        raise SFTTrainingError("resume checkpoint identity mismatch")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise SFTTrainingError(f"resume checkpoint file mismatch: {name}")
    load_safetensors_checkpoint(model, directory / "model.safetensors")
    optimizer.load_state_dict(
        torch.load(directory / "optimizer.pt", map_location=device, weights_only=False)
    )
    states = torch.load(
        directory / "distributed-state.pt", map_location="cpu", weights_only=False
    )
    local = next(state for state in states if state["rank"] == distributed.rank)
    _restore_rng(local["rng"], device)
    state = json.loads((directory / "trainer-state.json").read_text())
    return (
        state["global_step"],
        state["effective_assistant_target_tokens"],
        state.get("replay_input_tokens", 0),
    )


def _step_tokens(
    dataset: SFTDataset,
    config: SFTConfig,
    plan: TrainingPlan,
    step: int,
    block_order: np.ndarray,
) -> int:
    global_batch = config.world_size * config.micro_batch_size
    slot_base = (step % plan.steps_per_full_pass) * config.accumulation_steps
    result = 0
    for micro in range(config.accumulation_steps):
        start = (slot_base + micro) * global_batch
        if start < dataset.block_count:
            end = min(start + global_batch, dataset.block_count)
            result += int(dataset.target_counts[block_order[start:end]].sum())
    return result


def _is_replay_step(config: SFTConfig, completed_step: int) -> bool:
    """Return whether this completed SFT step also receives replay gradients."""
    return (
        config.replay_interval_steps is not None
        and completed_step % config.replay_interval_steps == 0
    )


def _expected_replay_input_tokens(
    config: SFTConfig, sequence_length: int, completed_steps: int
) -> int:
    if config.replay_interval_steps is None:
        return 0
    replay_steps = completed_steps // config.replay_interval_steps
    return (
        replay_steps
        * config.world_size
        * config.micro_batch_size
        * config.accumulation_steps
        * sequence_length
    )


def _load_replay_dataset(
    config: SFTConfig, distributed: DistributedContext, sequence_length: int
) -> ShardedTokenDataset | None:
    if config.replay_dataset is None:
        return None
    primary: ShardedTokenDataset | None = None
    payload: dict[str, Any] | None = None
    if distributed.is_main_process:
        try:
            primary = ShardedTokenDataset(
                config.replay_dataset, sequence_length=sequence_length
            )
            payload = {
                "ok": True,
                "manifest": primary.manifest,
                "manifest_sha256": primary.manifest_sha256,
            }
        except BaseException as error:
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = distributed.broadcast_object(payload)
    if not payload["ok"]:
        raise SFTTrainingError(f"rank-0 replay verification failed: {payload['error']}")
    dataset = primary or ShardedTokenDataset(
        config.replay_dataset,
        sequence_length=sequence_length,
        verified_manifest=payload["manifest"],
        manifest_sha256=payload["manifest_sha256"],
    )
    if dataset.manifest_sha256 != config.replay_manifest_sha256:
        raise SFTTrainingError("replay dataset manifest SHA-256 does not match config")
    return dataset


def run(args: argparse.Namespace, distributed: DistributedContext) -> int:
    root = args.project_root.resolve()
    config = load_sft_config(args.config, root)
    if distributed.world_size != config.world_size:
        raise SFTTrainingError(
            "runtime world size does not match the frozen SFT config"
        )
    if config.status == "release" and args.max_steps is not None:
        raise SFTTrainingError("release SFT rejects --max-steps")
    if config.status == "release" and args.repeat_first_batch:
        raise SFTTrainingError("release SFT rejects --repeat-first-batch")
    primary_dataset: SFTDataset | None = None
    dataset_payload: dict[str, Any] | None = None
    if distributed.is_main_process:
        try:
            primary_dataset = SFTDataset(config.dataset)
            dataset_payload = {
                "ok": True,
                "manifest": primary_dataset.manifest,
                "target_counts": primary_dataset.target_counts.tolist(),
            }
        except BaseException as error:
            dataset_payload = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    dataset_payload = distributed.broadcast_object(dataset_payload)
    if not dataset_payload["ok"]:
        raise SFTTrainingError(
            f"rank-0 dataset verification failed: {dataset_payload['error']}"
        )
    dataset = (
        primary_dataset
        if primary_dataset is not None
        else SFTDataset(
            config.dataset,
            manifest=dataset_payload["manifest"],
            target_counts=np.asarray(dataset_payload["target_counts"], dtype=np.int64),
        )
    )
    dataset_sha = _sha256(config.dataset / "manifest.json")
    if dataset_sha != config.dataset_manifest_sha256:
        raise SFTTrainingError("dataset manifest SHA-256 does not match config")
    _verify_training_data_policy(dataset.manifest, config.required_data_policy)
    replay_dataset = _load_replay_dataset(config, distributed, dataset.sequence_length)
    plan = build_training_plan(dataset, config)
    block_order = _block_order(dataset.block_count, config.seed)
    block_order_sha256 = hashlib.sha256(block_order.tobytes()).hexdigest()
    target_steps = (
        plan.total_steps
        if args.max_steps is None
        else min(args.max_steps, plan.total_steps)
    )

    if config.base_checkpoint is not None:
        base_outcome: dict[str, Any] | None = None
        if distributed.is_main_process:
            try:
                _verify_base_checkpoint(
                    config.base_checkpoint,
                    expected_manifest_sha256=config.base_manifest_sha256,
                    expected_model_sha256=config.base_model_sha256,
                )
                base_outcome = {"ok": True}
            except BaseException as error:
                base_outcome = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        base_outcome = distributed.broadcast_object(base_outcome)
        if not base_outcome["ok"]:
            raise SFTTrainingError(
                f"rank-0 base checkpoint verification failed: {base_outcome['error']}"
            )
    audit = {
        "status": "passed",
        "mode": config.status,
        "dataset_manifest_sha256": dataset_sha,
        "eligible_examples": dataset.manifest["eligible_examples"],
        "seen_unique_examples_at_completion": dataset.manifest["eligible_examples"],
        "eligible_train_coverage_ratio": 1.0,
        "omitted_eligible_examples": 0,
        "source_coverage_ratio": {
            source: 1.0 for source in dataset.manifest["source_eligible_examples"]
        },
        "unique_assistant_target_tokens": plan.unique_target_tokens,
        "repeated_exposure_tokens": plan.repeated_target_tokens,
        "effective_loss_tokens": plan.effective_target_tokens,
        "total_steps": plan.total_steps,
        "steps_per_full_pass": plan.steps_per_full_pass,
        "base_checkpoint_manifest_sha256": config.base_manifest_sha256,
        "base_model_sha256": config.base_model_sha256,
        "base_validation_status": config.base_validation_status,
        "sft_evaluation_status": "deferred",
        "replay_dataset_manifest_sha256": config.replay_manifest_sha256,
        "replay_interval_steps": config.replay_interval_steps,
        "replay_loss_weight": config.replay_loss_weight,
        "planned_replay_input_tokens": (
            _expected_replay_input_tokens(
                config, dataset.sequence_length, plan.total_steps
            )
        ),
        "ddp_static_graph": config.replay_dataset is None,
        "sft_block_order": "deterministic-permutation-v1",
        "sft_block_order_seed": config.seed,
        "sft_block_order_sha256": block_order_sha256,
    }
    if args.audit_only:
        if distributed.is_main_process:
            if args.audit_report is not None:
                args.audit_report.parent.mkdir(parents=True, exist_ok=True)
                args.audit_report.write_text(
                    json.dumps(audit, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(audit, indent=2, sort_keys=True))
        return 0

    device = distributed.device("cuda")
    random.seed(config.seed + distributed.rank)
    np.random.seed(config.seed + distributed.rank)
    torch.manual_seed(config.seed + distributed.rank)
    torch.cuda.manual_seed(config.seed + distributed.rank)
    model_config = load_model_config(config.model_config)
    model = AtomLLM(model_config).to(device=device, dtype=torch.float32)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != config.expected_parameters
    ):
        raise SFTTrainingError("model parameter count mismatch")
    # DDP synchronizes parameters from rank 0 during construction. Loading the
    # same multi-gigabyte checkpoint independently on every rank only adds
    # page faults and makes non-zero ranks spin at the DDP initialization gate.
    if config.base_checkpoint is not None and (
        not distributed.is_distributed or distributed.is_main_process
    ):
        load_safetensors_checkpoint(model, config.base_checkpoint / "model.safetensors")
    training_model: torch.nn.Module = (
        DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=config.ddp_bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=config.replay_dataset is None,
        )
        if distributed.is_distributed
        else model
    )
    optimizer = _optimizer(model, config)
    config_sha = _sha256(args.config)
    run_dir = args.output_dir / args.run_id
    if args.resume:
        if not run_dir.is_dir():
            raise SFTTrainingError("resume run directory does not exist")
        completed_steps, effective_tokens, replay_input_tokens = _load_resume(
            model, optimizer, run_dir, config_sha, dataset_sha, distributed, device
        )
        restored_from_step: int | None = completed_steps
    else:
        if distributed.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "checkpoints").mkdir()
            (run_dir / "reports").mkdir()
            shutil.copy2(args.config, run_dir / "sft-config.yaml")
            (run_dir / "launch-manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(UTC).isoformat(),
                        "git_commit": _git(root, "rev-parse", "HEAD"),
                        "git_dirty": bool(_git(root, "status", "--porcelain")),
                        "base_checkpoint": None
                        if config.base_checkpoint is None
                        else str(config.base_checkpoint),
                        "release_plan": audit,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        distributed.barrier()
        completed_steps = 0
        effective_tokens = 0
        replay_input_tokens = 0
        restored_from_step = None

    starting_effective_tokens = effective_tokens
    starting_replay_input_tokens = replay_input_tokens
    replay_iterator = (
        None
        if replay_dataset is None
        else ResumableShardedBatchIterator(
            replay_dataset,
            batch_size=config.micro_batch_size,
            seed=config.replay_seed,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
    )
    completed_replay_micro_batches = (
        0
        if config.replay_interval_steps is None
        else completed_steps // config.replay_interval_steps * config.accumulation_steps
    )
    if replay_iterator is not None:
        for _ in range(completed_replay_micro_batches):
            replay_iterator.next_batch()
        expected_replay_tokens = _expected_replay_input_tokens(
            config, dataset.sequence_length, completed_steps
        )
        if replay_input_tokens != expected_replay_tokens:
            raise SFTTrainingError("resume replay token count is inconsistent")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics_path = run_dir / "reports" / "metrics.jsonl"
    if distributed.is_main_process and restored_from_step is not None:
        with (run_dir / "reports" / "resume-events.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "restored_from_step": restored_from_step,
                        "target_step": target_steps,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    model.train()
    for step in range(completed_steps, target_steps):
        logical_step = 0 if args.repeat_first_batch else step
        replay_this_step = _is_replay_step(config, step + 1)
        learning_rate = _lr(config, plan, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        global_step_targets = _step_tokens(
            dataset, config, plan, logical_step, block_order
        )
        weighted_loss = torch.zeros((), device=device)
        replay_loss_sum = torch.zeros((), device=device)
        global_batch = config.world_size * config.micro_batch_size
        slot_base = (
            logical_step % plan.steps_per_full_pass
        ) * config.accumulation_steps
        for micro in range(config.accumulation_steps):
            global_start = (slot_base + micro) * global_batch
            local_start = global_start + distributed.rank * config.micro_batch_size
            slots = [local_start + offset for offset in range(config.micro_batch_size)]
            valid = [slot < dataset.block_count for slot in slots]
            indices = [
                int(block_order[slot]) if flag else 0
                for slot, flag in zip(slots, valid, strict=True)
            ]
            inputs, labels, local_targets = dataset.batch(indices, valid)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_weights = None
            if config.eot_loss_weight != 1.0:
                label_weights = torch.ones_like(labels, dtype=torch.float32)
                label_weights.masked_fill_(
                    labels == EOT_ID,
                    config.eot_loss_weight,
                )
            segment_ids = (
                _packed_segment_ids(inputs)
                if config.isolate_packed_conversations
                else None
            )
            sync = not replay_this_step and micro == config.accumulation_steps - 1
            context = (
                nullcontext()
                if sync or not isinstance(training_model, DistributedDataParallel)
                else training_model.no_sync()
            )
            with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = training_model(
                    inputs,
                    segment_ids=segment_ids,
                    labels=labels,
                    label_weights=label_weights,
                    loss_chunk_size=config.loss_chunk_size,
                    gradient_checkpointing=config.gradient_checkpointing,
                    checkpoint_segment_layers=config.checkpoint_segment_layers,
                    checkpoint_interval_segments=config.checkpoint_interval_segments,
                )
                if output.loss is None or not torch.isfinite(output.loss):
                    raise SFTTrainingError("SFT loss is missing or non-finite")
                scale = local_targets * distributed.world_size / global_step_targets
                (output.loss * scale).backward()
                weighted_loss += output.loss.detach() * local_targets
        if replay_this_step:
            if replay_iterator is None:
                raise SFTTrainingError("replay schedule has no replay dataset")
            for micro in range(config.accumulation_steps):
                replay_inputs = replay_iterator.next_batch().to(
                    device, non_blocking=True
                )
                sync = micro == config.accumulation_steps - 1
                context = (
                    nullcontext()
                    if sync or not isinstance(training_model, DistributedDataParallel)
                    else training_model.no_sync()
                )
                with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    replay_output = training_model(
                        replay_inputs,
                        labels=replay_inputs,
                        loss_chunk_size=config.loss_chunk_size,
                        gradient_checkpointing=config.gradient_checkpointing,
                        checkpoint_segment_layers=config.checkpoint_segment_layers,
                        checkpoint_interval_segments=(
                            config.checkpoint_interval_segments
                        ),
                    )
                    if replay_output.loss is None or not torch.isfinite(
                        replay_output.loss
                    ):
                        raise SFTTrainingError("replay loss is missing or non-finite")
                    (
                        replay_output.loss
                        * config.replay_loss_weight
                        / config.accumulation_steps
                    ).backward()
                    replay_loss_sum += replay_output.loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_gradient_norm
        )
        if not torch.isfinite(norm):
            raise SFTTrainingError("SFT gradient norm is non-finite")
        optimizer.step()
        completed = step + 1
        effective_tokens += global_step_targets
        if replay_this_step:
            replay_input_tokens = _expected_replay_input_tokens(
                config, dataset.sequence_length, completed
            )
        if distributed.is_distributed:
            torch.distributed.all_reduce(weighted_loss)
            torch.distributed.all_reduce(replay_loss_sum)
        loss_value = float(weighted_loss / global_step_targets)
        replay_loss_value = (
            float(replay_loss_sum / (config.world_size * config.accumulation_steps))
            if replay_this_step
            else None
        )
        if distributed.is_main_process:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "elapsed_training_seconds": time.perf_counter() - started,
                            "effective_assistant_target_tokens": effective_tokens,
                            "learning_rate": learning_rate,
                            "loss": loss_value,
                            "replay_loss": replay_loss_value,
                            "replay_input_tokens": replay_input_tokens,
                            "run_assistant_target_tokens": (
                                effective_tokens - starting_effective_tokens
                            ),
                            "step": completed,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if distributed.is_main_process and (
            completed == 1 or completed % 5 == 0 or completed == target_steps
        ):
            print(
                json.dumps(
                    {
                        "step": completed,
                        "loss": loss_value,
                        "replay_loss": replay_loss_value,
                        "replay_input_tokens": replay_input_tokens,
                        "learning_rate": learning_rate,
                        "effective_assistant_target_tokens": effective_tokens,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if completed % config.checkpoint_every == 0 or completed == target_steps:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                run_dir=run_dir,
                step=completed,
                effective_tokens=effective_tokens,
                replay_input_tokens=replay_input_tokens,
                config_sha=config_sha,
                dataset_sha=dataset_sha,
                distributed=distributed,
                device=device,
                milestone=completed == plan.total_steps,
                keep_last=config.keep_last,
            )

    if distributed.is_main_process:
        elapsed_seconds = time.perf_counter() - started
        trained_tokens = effective_tokens - starting_effective_tokens
        formal_completion = (
            config.status == "release" and target_steps == plan.total_steps
        )
        report: dict[str, Any] = {
            "status": "passed" if formal_completion else "smoke-complete",
            "mode": config.status,
            "completed_steps": target_steps,
            "elapsed_seconds": elapsed_seconds,
            "trained_assistant_target_tokens": trained_tokens,
            "trained_replay_input_tokens": (
                replay_input_tokens - starting_replay_input_tokens
            ),
            "assistant_target_tokens_per_second": (
                trained_tokens / elapsed_seconds if elapsed_seconds else None
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "final_effective_assistant_target_tokens": effective_tokens,
            "final_replay_input_tokens": replay_input_tokens,
            "formal_completion_reached": formal_completion,
            "restored_from_step": restored_from_step,
        }
        if metrics_path.is_file():
            metrics = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            report["initial_loss"] = metrics[0]["loss"]
            report["final_loss"] = metrics[-1]["loss"]
            final_metrics = metrics[-1]
            if final_metrics.get("elapsed_training_seconds"):
                report["compute_assistant_target_tokens_per_second"] = (
                    final_metrics["run_assistant_target_tokens"]
                    / final_metrics["elapsed_training_seconds"]
                )
        if formal_completion:
            report.update(audit)
        else:
            report["release_plan"] = audit
            report["eligible_train_coverage_ratio"] = None
            report["seen_unique_examples_at_completion"] = None
            report["omitted_eligible_examples"] = None
        (run_dir / "reports" / "training-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        if formal_completion:
            (run_dir / "COMPLETED").write_text(f"{config.name}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AtomLLM with assistant-only SFT loss."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/training-runs")
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Write the rank-0 launch audit JSON (requires --audit-only).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Smoke configs only; release configs reject this option.",
    )
    parser.add_argument(
        "--repeat-first-batch",
        action="store_true",
        help="Smoke configs only; repeatedly overfit the first global batch.",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_report is not None and not args.audit_only:
        raise SFTTrainingError("--audit-report requires --audit-only")
    preview = load_sft_config(args.config, args.project_root.resolve())
    distributed = DistributedContext.initialize(
        DistributedConfig(enabled=preview.world_size > 1, backend="nccl")
    )
    try:
        return run(args, distributed)
    finally:
        distributed.close()


if __name__ == "__main__":
    raise SystemExit(main())
