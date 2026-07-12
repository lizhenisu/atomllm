"""Audit formal train/validation token shards and their resumable data cursor."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from atomllm.training.config import file_sha256
from atomllm.training.data import (
    ResumableShardedBatchIterator,
    ShardedTokenDataset,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/training/formal-token-shard-audit-v2.yaml")


class FormalTokenShardAuditError(RuntimeError):
    """Raised when a formal token-shard release check fails."""


@dataclass(frozen=True, slots=True)
class FormalTokenShardAuditConfig:
    name: str
    train_dir: Path
    validation_dir: Path
    split_manifest: Path
    split_manifest_sha256: str
    data_audit_manifest: Path
    data_audit_manifest_sha256: str
    sequence_length: int
    batch_size: int
    sampler_seed: int
    resume_probe_batches: int
    output_report: Path


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FormalTokenShardAuditError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FormalTokenShardAuditError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise FormalTokenShardAuditError(f"{field} must be a positive integer")
    return value


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalTokenShardAuditError(f"{field} must be 64 lowercase hex digits")
    return value


def load_formal_token_shard_audit_config(
    path: str | Path = DEFAULT_CONFIG,
) -> FormalTokenShardAuditConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FormalTokenShardAuditError(
            f"cannot read config: {config_path}"
        ) from error
    expected = {
        "schema_version",
        "name",
        "train_dir",
        "validation_dir",
        "split_manifest",
        "split_manifest_sha256",
        "data_audit_manifest",
        "data_audit_manifest_sha256",
        "sequence_length",
        "batch_size",
        "sampler_seed",
        "resume_probe_batches",
        "output_report",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise FormalTokenShardAuditError("token-shard audit config fields are invalid")
    if value["schema_version"] != SCHEMA_VERSION:
        raise FormalTokenShardAuditError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise FormalTokenShardAuditError("name must be a non-empty string")
    seed = value["sampler_seed"]
    if type(seed) is not int or seed < 0:
        raise FormalTokenShardAuditError("sampler_seed must be non-negative")
    return FormalTokenShardAuditConfig(
        name=value["name"],
        train_dir=_safe_path(value["train_dir"], "train_dir"),
        validation_dir=_safe_path(value["validation_dir"], "validation_dir"),
        split_manifest=_safe_path(value["split_manifest"], "split_manifest"),
        split_manifest_sha256=_sha(
            value["split_manifest_sha256"], "split_manifest_sha256"
        ),
        data_audit_manifest=_safe_path(
            value["data_audit_manifest"], "data_audit_manifest"
        ),
        data_audit_manifest_sha256=_sha(
            value["data_audit_manifest_sha256"], "data_audit_manifest_sha256"
        ),
        sequence_length=_positive_int(value["sequence_length"], "sequence_length"),
        batch_size=_positive_int(value["batch_size"], "batch_size"),
        sampler_seed=seed,
        resume_probe_batches=_positive_int(
            value["resume_probe_batches"], "resume_probe_batches"
        ),
        output_report=_safe_path(value["output_report"], "output_report"),
    )


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise FormalTokenShardAuditError(f"{label} is missing or has SHA drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalTokenShardAuditError(f"{label} must be an object")
    return value


def _verify_source_lineage(
    token_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    split: str,
) -> None:
    expected = split_manifest.get("shards", {}).get(split)
    actual = token_manifest.get("shards")
    if not isinstance(expected, list) or not isinstance(actual, list):
        raise FormalTokenShardAuditError(f"{split} shard lists are invalid")
    if len(actual) != len(expected):
        raise FormalTokenShardAuditError(f"{split} shard count mismatch")
    for index, (token_shard, source_shard) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if (
            token_shard.get("source_name") != source_shard.get("name")
            or token_shard.get("source_sha256") != source_shard.get("sha256")
            or token_shard.get("document_count") != source_shard.get("record_count")
        ):
            raise FormalTokenShardAuditError(
                f"{split} source lineage mismatch at shard {index}"
            )


def _probe_resume(
    dataset: ShardedTokenDataset,
    *,
    batch_size: int,
    seed: int,
    batches: int,
) -> None:
    uninterrupted = ResumableShardedBatchIterator(
        dataset, batch_size=batch_size, seed=seed
    )
    for _ in range(batches):
        uninterrupted.next_batch()
    state = uninterrupted.state()
    expected = [uninterrupted.next_batch() for _ in range(batches)]
    resumed = ResumableShardedBatchIterator(dataset, batch_size=batch_size, seed=seed)
    resumed.restore(state)
    actual = [resumed.next_batch() for _ in range(batches)]
    if any(
        not torch.equal(left, right)
        for left, right in zip(expected, actual, strict=True)
    ):
        raise FormalTokenShardAuditError("resumed batches are not token-identical")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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


def audit_formal_token_shards(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_formal_token_shard_audit_config(root / config_path)
    split_manifest = _load_bound_json(
        root / config.split_manifest,
        config.split_manifest_sha256,
        "split manifest",
    )
    data_audit = _load_bound_json(
        root / config.data_audit_manifest,
        config.data_audit_manifest_sha256,
        "data audit manifest",
    )
    if data_audit.get("training_eligible") is not True:
        raise FormalTokenShardAuditError("source data is not training eligible")
    train_dataset = ShardedTokenDataset(
        root / config.train_dir, sequence_length=config.sequence_length
    )
    validation_dataset = ShardedTokenDataset(
        root / config.validation_dir, sequence_length=config.sequence_length
    )
    train_manifest = train_dataset.manifest
    validation_manifest = validation_dataset.manifest
    if train_manifest.get("split") != "train":
        raise FormalTokenShardAuditError("train artifact has wrong split")
    if validation_manifest.get("split") != "validation":
        raise FormalTokenShardAuditError("validation artifact has wrong split")
    for manifest in (train_manifest, validation_manifest):
        if manifest.get("encode_special_tokens_as_text") is not True:
            raise FormalTokenShardAuditError("unsafe special-token encoding policy")
    if train_manifest["document_count"] != data_audit["counts"]["train"]:
        raise FormalTokenShardAuditError("train document count mismatch")
    if validation_manifest["document_count"] != data_audit["counts"]["validation"]:
        raise FormalTokenShardAuditError("validation document count mismatch")
    _verify_source_lineage(train_manifest, split_manifest, "train")
    _verify_source_lineage(validation_manifest, split_manifest, "validation")
    if train_manifest["tokenizer"] != validation_manifest["tokenizer"]:
        raise FormalTokenShardAuditError("train and validation tokenizer differ")

    first_shard_blocks = (
        train_manifest["shards"][0]["token_count"] // config.sequence_length
    )
    if first_shard_blocks <= 0 or first_shard_blocks >= len(train_dataset):
        raise FormalTokenShardAuditError("cannot probe a train shard boundary")
    if train_dataset[first_shard_blocks - 1].numel() != config.sequence_length:
        raise FormalTokenShardAuditError("last block before shard boundary is invalid")
    if train_dataset[first_shard_blocks].numel() != config.sequence_length:
        raise FormalTokenShardAuditError("first block after shard boundary is invalid")
    _probe_resume(
        train_dataset,
        batch_size=config.batch_size,
        seed=config.sampler_seed,
        batches=config.resume_probe_batches,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "split_manifest_sha256": config.split_manifest_sha256,
        "data_audit_manifest_sha256": config.data_audit_manifest_sha256,
        "train_manifest_sha256": train_dataset.manifest_sha256,
        "validation_manifest_sha256": validation_dataset.manifest_sha256,
        "tokenizer": train_manifest["tokenizer"],
        "train_document_count": train_manifest["document_count"],
        "validation_document_count": validation_manifest["document_count"],
        "train_token_count": train_manifest["token_count"],
        "validation_token_count": validation_manifest["token_count"],
        "train_sequence_count": len(train_dataset),
        "validation_sequence_count": len(validation_dataset),
        "sequence_length": config.sequence_length,
        "checks": {
            "artifact_hashes": True,
            "source_lineage": True,
            "document_counts": True,
            "tokenizer_match": True,
            "special_tokens_encoded_as_text": True,
            "cross_shard_read": True,
            "data_cursor_exact_resume": True,
        },
        "formal_training_eligible": True,
        "passed": True,
    }
    _write_json_atomic(root / config.output_report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_formal_token_shards(args.config, project_root=args.project_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
