"""Build a deterministic, low-memory tokenizer corpus from formal train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from atomllm.data.schema import CanonicalDocument
from atomllm.tokenizer.config import EXPECTED_SPECIAL_TOKENS


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_SELECTION_VERSION = "stable-document-id-threshold-v1"
DEFAULT_SPLIT_DIR = Path("artifacts/data/formal-70g/split-v1")
DEFAULT_AUDIT_DIR = Path("artifacts/data/formal-70g/audit-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizer-snapshots/formal-70g-v4")
DEFAULT_TOKENIZER_OUTPUT_DIR = Path("artifacts/tokenizers/atom-tokenizer-formal-v4")
DEFAULT_CONFIG = Path("configs/tokenizer/formal-snapshot-v1.yaml")


class FormalTokenizerSnapshotError(RuntimeError):
    """Raised when a formal tokenizer snapshot is incomplete or untraceable."""


@dataclass(frozen=True, slots=True)
class FormalSnapshotConfig:
    """Complete formal snapshot recipe loaded from YAML."""

    sample_ratio: float
    split_dir: Path
    audit_dir: Path
    output_dir: Path
    tokenizer_output_dir: Path
    progress_interval_seconds: float


def _config_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FormalTokenizerSnapshotError(f"{field_name} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FormalTokenizerSnapshotError(f"{field_name} must be a safe relative path")
    return path


def load_formal_snapshot_config(
    path: str | Path = DEFAULT_CONFIG,
) -> FormalSnapshotConfig:
    """Load a complete snapshot recipe without implicit field defaults."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"formal tokenizer snapshot config not found: {config_path}"
        )
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise FormalTokenizerSnapshotError(f"invalid snapshot YAML: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormalTokenizerSnapshotError("snapshot config must be a mapping")
    required = {
        "schema_version",
        "sample_ratio",
        "split_dir",
        "audit_dir",
        "output_dir",
        "tokenizer_output_dir",
        "progress_interval_seconds",
    }
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if unknown:
        raise FormalTokenizerSnapshotError(
            f"snapshot config has unknown fields: {', '.join(unknown)}"
        )
    if missing:
        raise FormalTokenizerSnapshotError(
            f"snapshot config missing required fields: {', '.join(missing)}"
        )
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise FormalTokenizerSnapshotError(
            f"schema_version must be {SNAPSHOT_SCHEMA_VERSION}"
        )
    ratio = value["sample_ratio"]
    _ratio_threshold(ratio)
    interval = value["progress_interval_seconds"]
    if type(interval) not in {int, float} or float(interval) <= 0:
        raise FormalTokenizerSnapshotError("progress_interval_seconds must be positive")
    return FormalSnapshotConfig(
        sample_ratio=float(ratio),
        split_dir=_config_path(value["split_dir"], "split_dir"),
        audit_dir=_config_path(value["audit_dir"], "audit_dir"),
        output_dir=_config_path(value["output_dir"], "output_dir"),
        tokenizer_output_dir=_config_path(
            value["tokenizer_output_dir"],
            "tokenizer_output_dir",
        ),
        progress_interval_seconds=float(interval),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalTokenizerSnapshotError(f"cannot read {context}: {path}") from error
    if not isinstance(value, dict):
        raise FormalTokenizerSnapshotError(f"{context} must be a JSON object")
    return value


def _canonical_json(value: dict[str, Any], *, pretty: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _resolve(root: Path, path: str | Path, field_name: str) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not resolved.is_relative_to(root):
        raise FormalTokenizerSnapshotError(
            f"{field_name} resolves outside project root"
        )
    return resolved


def _relative(root: Path, path: Path, field_name: str) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise FormalTokenizerSnapshotError(
            f"{field_name} resolves outside project root"
        ) from error


def _ratio_threshold(sample_ratio: float) -> int:
    if type(sample_ratio) not in {int, float} or not 0 < float(sample_ratio) <= 1:
        raise FormalTokenizerSnapshotError("sample_ratio must be a number in (0, 1]")
    return int(float(sample_ratio) * (1 << 256))


def _is_selected(document_id: str, threshold: int) -> bool:
    identity = f"{SNAPSHOT_SELECTION_VERSION}\0{document_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest(), "big") < threshold


def _validate_inputs(
    split_dir: Path,
    audit_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    split_manifest_path = split_dir / "manifest.json"
    audit_manifest_path = audit_dir / "manifest.json"
    if not split_manifest_path.is_file() or not audit_manifest_path.is_file():
        raise FormalTokenizerSnapshotError("formal split or audit manifest is missing")
    split_manifest = _read_json(split_manifest_path, "formal split manifest")
    audit_manifest = _read_json(audit_manifest_path, "formal audit manifest")
    if split_manifest.get("training_eligible") is not True:
        raise FormalTokenizerSnapshotError("formal split is not training eligible")
    if audit_manifest.get("training_eligible") is not True:
        raise FormalTokenizerSnapshotError("formal audit is not training eligible")
    checks = audit_manifest.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise FormalTokenizerSnapshotError("formal audit checks did not all pass")
    split_sha256 = _sha256(split_manifest_path)
    provenance = audit_manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("split") != split_sha256:
        raise FormalTokenizerSnapshotError("formal audit does not match split manifest")
    shards = split_manifest.get("shards")
    if not isinstance(shards, dict) or not isinstance(shards.get("train"), list):
        raise FormalTokenizerSnapshotError(
            "formal split has invalid train shard metadata"
        )
    if not shards["train"]:
        raise FormalTokenizerSnapshotError("formal split has no train shards")
    return split_manifest, audit_manifest, split_sha256, _sha256(audit_manifest_path)


def _training_config(
    *,
    data_version_id: str,
    document_count: int,
    snapshot_sha256: str,
    snapshot_path: str,
    tokenizer_output_dir: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "atom-tokenizer-formal-v1",
        "status": "release",
        "training_eligible": True,
        "model_max_length": 8192,
        "algorithm": {
            "model_type": "byte_level_bpe",
            "vocab_size": 32000,
            "normalization": "nfc",
            "pre_tokenizer": "byte_level",
            "decoder": "byte_level",
            "min_frequency": 2,
            "dropout": 0.0,
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": True,
            "byte_fallback": False,
            "fuse_unk": False,
            "ignore_merges": False,
            "max_token_length": 24,
        },
        "special_tokens": [
            {"id": token_id, "token": token, "purpose": purpose}
            for token_id, token, purpose in EXPECTED_SPECIAL_TOKENS
        ],
        "training_data": {
            "data_version_id": data_version_id,
            "split": "train",
            "document_count": document_count,
            "expected_sha256": snapshot_sha256,
            "input_path": snapshot_path,
        },
        "evaluation": {
            "roundtrip_required": True,
            "max_unknown_rate": 0.0,
            "suites": [
                "zh-Hans",
                "en",
                "zh-Hant",
                "ja",
                "code",
                "math",
                "digits",
                "whitespace",
            ],
        },
        "output_dir": tokenizer_output_dir,
    }


def _validate_existing(
    output_dir: Path,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    snapshot_path = output_dir / "documents.jsonl"
    config_path = output_dir / "tokenizer-training.yaml"
    completed_path = output_dir / "COMPLETED"
    if not all(
        path.is_file()
        for path in (manifest_path, snapshot_path, config_path, completed_path)
    ):
        raise FormalTokenizerSnapshotError("existing tokenizer snapshot is incomplete")
    manifest = _read_json(manifest_path, "tokenizer snapshot manifest")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise FormalTokenizerSnapshotError(
            "existing tokenizer snapshot uses different input"
        )
    snapshot = manifest.get("snapshot")
    if not isinstance(snapshot, dict):
        raise FormalTokenizerSnapshotError(
            "existing tokenizer snapshot has invalid metadata"
        )
    if _sha256(snapshot_path) != snapshot.get("sha256"):
        raise FormalTokenizerSnapshotError(
            "existing tokenizer snapshot SHA-256 mismatch"
        )
    if _sha256(config_path) != manifest.get("training_config_sha256"):
        raise FormalTokenizerSnapshotError(
            "existing tokenizer training config SHA-256 mismatch"
        )
    if completed_path.read_text(encoding="utf-8") != (
        f"{_sha256(manifest_path)}  manifest.json\n"
    ):
        raise FormalTokenizerSnapshotError(
            "existing tokenizer snapshot COMPLETED marker is invalid"
        )
    return manifest


def build_formal_tokenizer_snapshot(
    *,
    project_root: str | Path = ".",
    split_dir: str | Path,
    audit_dir: str | Path,
    output_dir: str | Path,
    tokenizer_output_dir: str | Path,
    sample_ratio: float,
    progress_interval_seconds: float,
) -> dict[str, Any]:
    """Create an immutable, stratified-by-hash tokenizer training snapshot."""
    root = Path(project_root).resolve()
    split_path = _resolve(root, split_dir, "split_dir")
    audit_path = _resolve(root, audit_dir, "audit_dir")
    output_path = _resolve(root, output_dir, "output_dir")
    tokenizer_output_path = _resolve(root, tokenizer_output_dir, "tokenizer_output_dir")
    threshold = _ratio_threshold(sample_ratio)
    if progress_interval_seconds is not None and (
        type(progress_interval_seconds) not in {int, float}
        or float(progress_interval_seconds) <= 0
    ):
        raise FormalTokenizerSnapshotError(
            "progress_interval_seconds must be positive or None"
        )
    split_manifest, _audit_manifest, split_sha256, audit_sha256 = _validate_inputs(
        split_path, audit_path
    )
    identity = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_version": "formal-tokenizer-snapshot-v1",
        "selection_version": SNAPSHOT_SELECTION_VERSION,
        "sample_ratio": float(sample_ratio),
        "split_manifest_sha256": split_sha256,
        "audit_manifest_sha256": audit_sha256,
    }
    existing = _validate_existing(output_path, identity)
    if existing is not None:
        return existing

    if output_path.exists():
        raise FormalTokenizerSnapshotError("snapshot output directory already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        documents_path = temporary_dir / "documents.jsonl"
        source_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        content_counts: Counter[str] = Counter()
        strata_counts: Counter[str] = Counter()
        input_document_count = 0
        output_document_count = 0
        input_bytes = 0
        output_bytes = 0
        input_shards: list[dict[str, Any]] = []
        train_shards = split_manifest["shards"]["train"]
        total_input_bytes = sum(
            (split_path / "train" / "shards" / str(shard.get("name", "")))
            .stat()
            .st_size
            for shard in train_shards
            if isinstance(shard, dict)
        )
        started_at = time.monotonic()
        last_progress_at = started_at
        if progress_interval_seconds is not None:
            print(
                "[tokenizer-snapshot] start "
                f"target_ratio={float(sample_ratio):.2%} "
                f"input={total_input_bytes / (1024**3):.3f}GiB "
                f"shards={len(train_shards)}",
                file=sys.stderr,
                flush=True,
            )
        with documents_path.open("wb") as output_handle:
            for shard_index, shard in enumerate(train_shards, start=1):
                if not isinstance(shard, dict):
                    raise FormalTokenizerSnapshotError(
                        "train shard metadata is invalid"
                    )
                name = shard.get("name")
                expected_sha256 = shard.get("sha256")
                expected_count = shard.get("record_count")
                if (
                    not isinstance(name, str)
                    or not isinstance(expected_sha256, str)
                    or type(expected_count) is not int
                ):
                    raise FormalTokenizerSnapshotError(
                        "train shard metadata is invalid"
                    )
                shard_path = split_path / "train" / "shards" / name
                if not shard_path.is_file():
                    raise FormalTokenizerSnapshotError(
                        f"train shard is missing: {name}"
                    )
                digest = hashlib.sha256()
                line_count = 0
                with shard_path.open("rb") as input_handle:
                    for raw_line in input_handle:
                        digest.update(raw_line)
                        line_count += 1
                        input_bytes += len(raw_line)
                        try:
                            line = raw_line.decode("utf-8")
                        except UnicodeDecodeError as error:
                            raise FormalTokenizerSnapshotError(
                                f"train shard is not UTF-8: {name}"
                            ) from error
                        document = CanonicalDocument.from_json_line(line)
                        input_document_count += 1
                        if not _is_selected(document.document_id, threshold):
                            continue
                        output_handle.write(raw_line)
                        output_document_count += 1
                        output_bytes += len(raw_line)
                        source_counts[document.source_id] += 1
                        language_counts[document.language] += 1
                        content_counts[document.content_type] += 1
                        strata_counts[
                            f"{document.source_id}|{document.language}|{document.content_type}"
                        ] += 1
                        if (
                            progress_interval_seconds is not None
                            and input_document_count % 10_000 == 0
                            and time.monotonic() - last_progress_at
                            >= float(progress_interval_seconds)
                        ):
                            elapsed = time.monotonic() - started_at
                            rate = input_bytes / elapsed if elapsed else 0.0
                            print(
                                "[tokenizer-snapshot] progress "
                                f"shard={shard_index}/{len(train_shards)} "
                                f"input={input_bytes / (1024**3):.3f}GiB/"
                                f"{total_input_bytes / (1024**3):.3f}GiB "
                                f"selected={output_document_count} "
                                f"rate={rate / (1024**2):.2f}MiB/s",
                                file=sys.stderr,
                                flush=True,
                            )
                            last_progress_at = time.monotonic()
                actual_sha256 = digest.hexdigest()
                if actual_sha256 != expected_sha256 or line_count != expected_count:
                    raise FormalTokenizerSnapshotError(
                        f"train shard integrity mismatch: {name}"
                    )
                input_shards.append(
                    {
                        "name": name,
                        "record_count": line_count,
                        "sha256": actual_sha256,
                    }
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if output_document_count == 0:
            raise FormalTokenizerSnapshotError(
                "sample_ratio selected no tokenizer documents"
            )
        snapshot_sha256 = _sha256(documents_path)
        snapshot_identity = {
            **identity,
            "document_count": output_document_count,
            "sha256": snapshot_sha256,
        }
        identity_sha256 = hashlib.sha256(
            _canonical_json(snapshot_identity, pretty=False).encode("utf-8")
        ).hexdigest()
        data_version_id = (
            f"data-formal-70g-tokenizer-snapshot-v1-{identity_sha256[:12]}"
        )
        training_config = _training_config(
            data_version_id=data_version_id,
            document_count=output_document_count,
            snapshot_sha256=snapshot_sha256,
            snapshot_path=_relative(
                root, output_path / "documents.jsonl", "snapshot_path"
            ),
            tokenizer_output_dir=_relative(
                root, tokenizer_output_path, "tokenizer_output_dir"
            ),
        )
        config_path = temporary_dir / "tokenizer-training.yaml"
        config_path.write_text(
            yaml.safe_dump(training_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest = {
            **identity,
            "data_version_id": data_version_id,
            "identity_sha256": identity_sha256,
            "input": {
                "split_dir": _relative(root, split_path, "split_dir"),
                "train_document_count": input_document_count,
                "train_bytes": input_bytes,
                "shards": input_shards,
            },
            "snapshot": {
                "file": "documents.jsonl",
                "document_count": output_document_count,
                "bytes": output_bytes,
                "document_ratio": round(
                    output_document_count / input_document_count, 8
                ),
                "byte_ratio": round(output_bytes / input_bytes, 8),
                "sha256": snapshot_sha256,
            },
            "distribution": {
                "source_counts": dict(sorted(source_counts.items())),
                "language_counts": dict(sorted(language_counts.items())),
                "content_type_counts": dict(sorted(content_counts.items())),
                "strata_counts": dict(sorted(strata_counts.items())),
            },
            "training_config": "tokenizer-training.yaml",
            "training_config_sha256": _sha256(config_path),
        }
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(
            f"{_canonical_json(manifest, pretty=True)}\n", encoding="utf-8"
        )
        (temporary_dir / "COMPLETED").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary_dir, output_path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic formal tokenizer training snapshot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Snapshot recipe YAML (default: {DEFAULT_CONFIG}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_formal_snapshot_config(args.config)
    manifest = build_formal_tokenizer_snapshot(
        split_dir=config.split_dir,
        audit_dir=config.audit_dir,
        output_dir=config.output_dir,
        tokenizer_output_dir=config.tokenizer_output_dir,
        sample_ratio=config.sample_ratio,
        progress_interval_seconds=config.progress_interval_seconds,
    )
    snapshot = manifest["snapshot"]
    print(
        "Formal tokenizer snapshot complete: "
        f"documents={snapshot['document_count']}, "
        f"bytes={snapshot['bytes']}, "
        f"ratio={snapshot['byte_ratio']:.2%}, "
        f"data_version_id={manifest['data_version_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
