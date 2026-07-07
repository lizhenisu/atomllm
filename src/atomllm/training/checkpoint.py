"""Transactional exact-resume checkpoints for AtomLLM training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from atomllm.model.checkpoint import (
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from atomllm.training.scheduler import SchedulerError
from atomllm.training.state import DataState, TrainerState
from atomllm.training.trainer import Trainer


CHECKPOINT_FORMAT_VERSION = 1
COMPLETE_CONTENT = "atomllm-checkpoint-complete-v1\n"
PAYLOAD_FILES = {
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
    "trainer_state.json",
    "data_state.json",
}
_CHECKPOINT_ID_PATTERN = re.compile(r"^step-[0-9]{9}$")
_LOGICAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CheckpointError(RuntimeError):
    """Raised when a training checkpoint cannot be saved or restored safely."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    run_id: str
    project_version: str
    git_commit: str
    git_dirty: bool
    tokenizer_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        if _LOGICAL_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a safe logical ID")
        if not self.project_version:
            raise ValueError("project_version must be non-empty")
        if not self.git_commit:
            raise ValueError("git_commit must be non-empty")
        if type(self.git_dirty) is not bool:
            raise TypeError("git_dirty must be a boolean")
        for field_name in ("tokenizer_sha256", "config_sha256"):
            if _SHA256_PATTERN.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(f"{field_name} must be 64 lowercase hex digits")


@dataclass(frozen=True, slots=True)
class SavedCheckpoint:
    checkpoint_id: str
    directory: Path
    manifest_sha256: str
    removed_checkpoint_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any, *, pretty: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(f"{_canonical_json(value, pretty=True)}\n", encoding="utf-8")
    _fsync_file(path)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read {context}: {path.name}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"{context} must be a JSON object")
    return value


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _torch_save(value: Any, path: Path) -> None:
    torch.save(value, path)
    _fsync_file(path)


def checkpoint_id(global_step: int) -> str:
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("global_step must be a positive integer")
    if global_step > 999_999_999:
        raise ValueError("global_step exceeds the checkpoint ID range")
    return f"step-{global_step:09d}"


def model_signature(trainer: Trainer) -> str:
    model = trainer.model
    signature = {
        "contract_version": 1,
        "model_config": asdict(model.config),
        "parameters": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in model.named_parameters(remove_duplicate=False)
        ],
        "tied_word_embeddings": (model.lm_head.weight is model.token_embeddings.weight),
    }
    return hashlib.sha256(
        _canonical_json(signature, pretty=False).encode("utf-8")
    ).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "format_version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != {
        "format_version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointError("RNG state has an invalid structure")
    if state["format_version"] != 1:
        raise CheckpointError("RNG state format version is unsupported")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise CheckpointError("CUDA RNG state must be a list")
    if cuda_states and not torch.cuda.is_available():
        raise CheckpointError(
            "checkpoint contains CUDA RNG state but CUDA is unavailable"
        )
    if cuda_states and len(cuda_states) != torch.cuda.device_count():
        raise CheckpointError("CUDA RNG state device count does not match")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def _file_records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "size_bytes": (directory / name).stat().st_size,
            "sha256": _sha256(directory / name),
        }
        for name in sorted(PAYLOAD_FILES)
    }


def _validate_manifest_structure(
    manifest: dict[str, Any],
    expected_checkpoint_id: str,
) -> None:
    required = {
        "format_version",
        "checkpoint_id",
        "run_id",
        "created_at",
        "global_step",
        "tokens_seen",
        "project_version",
        "git_commit",
        "git_dirty",
        "model_signature",
        "tokenizer_sha256",
        "config_sha256",
        "dataset_id",
        "dataset_manifest_sha256",
        "milestone",
        "files",
    }
    if set(manifest) != required:
        raise CheckpointError("checkpoint manifest fields are invalid")
    if manifest["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("checkpoint format version is unsupported")
    if manifest["checkpoint_id"] != expected_checkpoint_id:
        raise CheckpointError("checkpoint ID does not match its directory")
    if manifest["global_step"] != int(expected_checkpoint_id.removeprefix("step-")):
        raise CheckpointError("checkpoint global_step does not match its ID")
    if type(manifest["milestone"]) is not bool:
        raise CheckpointError("checkpoint milestone flag is invalid")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != PAYLOAD_FILES:
        raise CheckpointError("checkpoint payload file list is invalid")


def verify_checkpoint_directory(
    directory: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(directory)
    checkpoint_name = checkpoint_dir.name
    if _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_name) is None:
        raise CheckpointError("checkpoint directory has an invalid ID")
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"checkpoint directory not found: {checkpoint_name}")
    complete_path = checkpoint_dir / "COMPLETE"
    manifest_path = checkpoint_dir / "manifest.json"
    if (
        not complete_path.is_file()
        or complete_path.read_text(encoding="utf-8") != COMPLETE_CONTENT
    ):
        raise CheckpointError("checkpoint COMPLETE marker is missing or invalid")
    if not manifest_path.is_file():
        raise CheckpointError("checkpoint manifest is missing")
    manifest_sha256 = _sha256(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise CheckpointError("checkpoint manifest SHA-256 does not match")
    manifest = _read_json(manifest_path, "checkpoint manifest")
    _validate_manifest_structure(manifest, checkpoint_name)
    actual_payload_files = {
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    }
    if actual_payload_files != PAYLOAD_FILES:
        raise CheckpointError("checkpoint directory contains an unexpected payload")
    for name, metadata in manifest["files"].items():
        path = checkpoint_dir / name
        if not isinstance(metadata, dict) or set(metadata) != {
            "size_bytes",
            "sha256",
        }:
            raise CheckpointError(f"checkpoint metadata is invalid: {name}")
        if not path.is_file():
            raise CheckpointError(f"checkpoint payload is missing: {name}")
        if path.stat().st_size != metadata["size_bytes"]:
            raise CheckpointError(f"checkpoint payload size does not match: {name}")
        if _sha256(path) != metadata["sha256"]:
            raise CheckpointError(f"checkpoint payload SHA-256 does not match: {name}")
    return manifest


def _write_latest(
    checkpoints_dir: Path,
    checkpoint_name: str,
    manifest_sha256: str,
) -> None:
    latest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_id": checkpoint_name,
        "manifest_sha256": manifest_sha256,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".latest.json.tmp-",
        dir=checkpoints_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_json(temporary_path, latest)
        os.replace(temporary_path, checkpoints_dir / "latest.json")
        _fsync_directory(checkpoints_dir)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_latest(checkpoints_dir: Path) -> tuple[str, str]:
    latest = _read_json(checkpoints_dir / "latest.json", "latest pointer")
    if set(latest) != {
        "format_version",
        "checkpoint_id",
        "manifest_sha256",
    }:
        raise CheckpointError("latest pointer fields are invalid")
    if latest["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("latest pointer format version is unsupported")
    checkpoint_name = latest["checkpoint_id"]
    manifest_sha256 = latest["manifest_sha256"]
    if (
        not isinstance(checkpoint_name, str)
        or _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_name) is None
    ):
        raise CheckpointError("latest pointer checkpoint ID is invalid")
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256_PATTERN.fullmatch(manifest_sha256) is None
    ):
        raise CheckpointError("latest pointer manifest SHA-256 is invalid")
    return checkpoint_name, manifest_sha256


def _prune_checkpoints(
    checkpoints_dir: Path,
    keep_last: int,
) -> tuple[str, ...]:
    candidates: list[tuple[int, Path, bool]] = []
    for path in checkpoints_dir.iterdir():
        if not path.is_dir() or _CHECKPOINT_ID_PATTERN.fullmatch(path.name) is None:
            continue
        try:
            manifest = verify_checkpoint_directory(path)
        except CheckpointError:
            continue
        candidates.append((manifest["global_step"], path, manifest["milestone"]))
    candidates.sort()
    recent = {path.name for _, path, _ in candidates[-keep_last:]}
    removed: list[str] = []
    for _, path, milestone in candidates:
        if path.name in recent or milestone:
            continue
        try:
            shutil.rmtree(path)
            removed.append(path.name)
        except OSError as error:
            warnings.warn(
                f"could not remove old checkpoint {path.name}: {error}",
                stacklevel=2,
            )
    if removed:
        _fsync_directory(checkpoints_dir)
    return tuple(removed)


def save_training_checkpoint(
    trainer: Trainer,
    checkpoints_dir: str | Path,
    identity: CheckpointIdentity,
    *,
    keep_last: int,
    milestone: bool = False,
) -> SavedCheckpoint:
    """Atomically save all state needed for exact training continuation."""
    if type(keep_last) is not int or keep_last <= 0:
        raise ValueError("keep_last must be a positive integer")
    trainer_state = trainer.trainer_state()
    data_state = trainer.data_iterator.state()
    checkpoint_name = checkpoint_id(trainer_state.global_step)
    root = Path(checkpoints_dir)
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / checkpoint_name
    if final_dir.exists():
        raise CheckpointError(f"checkpoint already exists: {checkpoint_name}")
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{checkpoint_name}.tmp-", dir=root))
    try:
        save_safetensors_checkpoint(
            trainer.model,
            temporary_dir / "model.safetensors",
        )
        _fsync_file(temporary_dir / "model.safetensors")
        _torch_save(trainer.optimizer.state_dict(), temporary_dir / "optimizer.pt")
        _torch_save(trainer.scheduler.state_dict(), temporary_dir / "scheduler.pt")
        _torch_save(capture_rng_state(), temporary_dir / "rng_state.pt")
        _write_json(
            temporary_dir / "trainer_state.json",
            trainer_state.to_mapping(),
        )
        _write_json(temporary_dir / "data_state.json", data_state.to_mapping())
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_id": checkpoint_name,
            "run_id": identity.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "global_step": trainer_state.global_step,
            "tokens_seen": trainer_state.tokens_seen,
            "project_version": identity.project_version,
            "git_commit": identity.git_commit,
            "git_dirty": identity.git_dirty,
            "model_signature": model_signature(trainer),
            "tokenizer_sha256": identity.tokenizer_sha256,
            "config_sha256": identity.config_sha256,
            "dataset_id": data_state.dataset_id,
            "dataset_manifest_sha256": data_state.dataset_manifest_sha256,
            "milestone": milestone,
            "files": _file_records(temporary_dir),
        }
        manifest_path = temporary_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        complete_path = temporary_dir / "COMPLETE"
        complete_path.write_text(COMPLETE_CONTENT, encoding="utf-8")
        _fsync_file(complete_path)
        _fsync_directory(temporary_dir)
        manifest_sha256 = _sha256(manifest_path)
        os.replace(temporary_dir, final_dir)
        _fsync_directory(root)
        verify_checkpoint_directory(
            final_dir,
            expected_manifest_sha256=manifest_sha256,
        )
        _write_latest(root, checkpoint_name, manifest_sha256)
        removed = _prune_checkpoints(root, keep_last)
        return SavedCheckpoint(
            checkpoint_id=checkpoint_name,
            directory=final_dir,
            manifest_sha256=manifest_sha256,
            removed_checkpoint_ids=removed,
        )
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _validate_compatibility(
    manifest: dict[str, Any],
    trainer: Trainer,
    identity: CheckpointIdentity,
) -> None:
    data_state = trainer.data_iterator.state()
    expected = {
        "run_id": identity.run_id,
        "project_version": identity.project_version,
        "model_signature": model_signature(trainer),
        "tokenizer_sha256": identity.tokenizer_sha256,
        "config_sha256": identity.config_sha256,
        "dataset_id": data_state.dataset_id,
        "dataset_manifest_sha256": data_state.dataset_manifest_sha256,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if manifest[key] != expected_value
    ]
    if mismatches:
        raise CheckpointError(
            f"checkpoint is incompatible: {', '.join(sorted(mismatches))}"
        )


def restore_training_checkpoint(
    trainer: Trainer,
    checkpoints_dir: str | Path,
    identity: CheckpointIdentity,
    *,
    selected_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Verify every byte, restore all training state, and restore RNG last."""
    root = Path(checkpoints_dir)
    expected_manifest_sha256: str | None = None
    if selected_checkpoint_id is None:
        checkpoint_name, expected_manifest_sha256 = _read_latest(root)
    else:
        checkpoint_name = selected_checkpoint_id
        if _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_name) is None:
            raise CheckpointError("selected checkpoint ID is invalid")
    checkpoint_dir = root / checkpoint_name
    manifest = verify_checkpoint_directory(
        checkpoint_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    _validate_compatibility(manifest, trainer, identity)

    trainer_state = TrainerState.from_mapping(
        _read_json(checkpoint_dir / "trainer_state.json", "trainer state")
    )
    data_state = DataState.from_mapping(
        _read_json(checkpoint_dir / "data_state.json", "data state")
    )
    load_safetensors_checkpoint(
        trainer.model,
        checkpoint_dir / "model.safetensors",
    )
    optimizer_state = torch.load(
        checkpoint_dir / "optimizer.pt",
        map_location=trainer.device,
        weights_only=False,
    )
    scheduler_state = torch.load(
        checkpoint_dir / "scheduler.pt",
        map_location="cpu",
        weights_only=False,
    )
    rng_state = torch.load(
        checkpoint_dir / "rng_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    trainer.optimizer.load_state_dict(optimizer_state)
    try:
        trainer.scheduler.load_state_dict(scheduler_state)
    except SchedulerError as error:
        raise CheckpointError(f"scheduler restore failed: {error}") from error
    trainer.restore_state(trainer_state, data_state)
    restore_rng_state(rng_state)
    return manifest
