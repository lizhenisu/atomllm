"""Stream a formal JSONL split into resumable token and document-index shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from atomllm.data.schema import CanonicalDocument
from atomllm.training.config import file_sha256


SCHEMA_VERSION = 1
FORMAT_VERSION = "document-bos-eos-sharded-v2"
DEFAULT_CONFIG = Path("configs/training/formal-token-shards-v2.yaml")
STATE_NAME = "state.json"
MANIFEST_NAME = "manifest.json"
COMPLETED_NAME = "COMPLETED"
_SHA256_LENGTH = 64
_GIB = 1024**3


class FormalTokenShardError(RuntimeError):
    """Raised when formal token-shard construction cannot continue safely."""


@dataclass(frozen=True, slots=True)
class FormalTokenShardConfig:
    name: str
    split_dir: Path
    split_manifest_sha256: str
    audit_manifest: Path
    audit_manifest_sha256: str
    tokenizer_version_manifest: Path
    tokenizer_version_manifest_sha256: str
    tokenizer_path: Path
    tokenizer_sha256: str
    output_dir: Path
    token_dtype: str
    max_rss_gib: float
    progress_interval_seconds: int
    workers: int
    input_split: str = "train"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalTokenShardError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise FormalTokenShardError(f"{label} must be an object")
    return value


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FormalTokenShardError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FormalTokenShardError(f"{field} must be a safe relative path")
    return path


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalTokenShardError(f"{field} must be 64 lowercase hex digits")
    return value


def load_formal_token_shard_config(
    path: str | Path = DEFAULT_CONFIG,
) -> FormalTokenShardConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FormalTokenShardError(f"cannot read config: {config_path}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormalTokenShardError("config must be a mapping with string keys")
    required = {
        "schema_version",
        "name",
        "split_dir",
        "split_manifest_sha256",
        "audit_manifest",
        "audit_manifest_sha256",
        "tokenizer_version_manifest",
        "tokenizer_version_manifest_sha256",
        "tokenizer_path",
        "tokenizer_sha256",
        "output_dir",
        "token_dtype",
        "max_rss_gib",
        "progress_interval_seconds",
        "workers",
    }
    optional = {"input_split"}
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise FormalTokenShardError(f"config missing fields: {', '.join(missing)}")
    if unknown:
        raise FormalTokenShardError(f"config has unknown fields: {', '.join(unknown)}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise FormalTokenShardError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise FormalTokenShardError("name must be a non-empty string")
    if value["token_dtype"] != "uint16-le":
        raise FormalTokenShardError("token_dtype must be uint16-le")
    input_split = value.get("input_split", "train")
    if input_split not in {"train", "validation"}:
        raise FormalTokenShardError("input_split must be train or validation")
    max_rss = value["max_rss_gib"]
    if type(max_rss) not in {int, float} or not 0 < float(max_rss) <= 9:
        raise FormalTokenShardError("max_rss_gib must be greater than 0 and at most 9")
    interval = value["progress_interval_seconds"]
    if type(interval) is not int or interval <= 0:
        raise FormalTokenShardError("progress_interval_seconds must be positive")
    workers = value["workers"]
    cpu_count = os.cpu_count() or 1
    if type(workers) is not int or not 1 <= workers <= cpu_count:
        raise FormalTokenShardError(f"workers must be between 1 and {cpu_count}")
    return FormalTokenShardConfig(
        name=value["name"],
        split_dir=_safe_path(value["split_dir"], "split_dir"),
        split_manifest_sha256=_sha(
            value["split_manifest_sha256"], "split_manifest_sha256"
        ),
        audit_manifest=_safe_path(value["audit_manifest"], "audit_manifest"),
        audit_manifest_sha256=_sha(
            value["audit_manifest_sha256"], "audit_manifest_sha256"
        ),
        tokenizer_version_manifest=_safe_path(
            value["tokenizer_version_manifest"], "tokenizer_version_manifest"
        ),
        tokenizer_version_manifest_sha256=_sha(
            value["tokenizer_version_manifest_sha256"],
            "tokenizer_version_manifest_sha256",
        ),
        tokenizer_path=_safe_path(value["tokenizer_path"], "tokenizer_path"),
        tokenizer_sha256=_sha(value["tokenizer_sha256"], "tokenizer_sha256"),
        output_dir=_safe_path(value["output_dir"], "output_dir"),
        token_dtype=value["token_dtype"],
        max_rss_gib=float(max_rss),
        progress_interval_seconds=interval,
        workers=workers,
        input_split=input_split,
    )


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)}\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _identity(config: FormalTokenShardConfig) -> dict[str, Any]:
    identity = {
        "format_version": FORMAT_VERSION,
        "name": config.name,
        "split_manifest_sha256": config.split_manifest_sha256,
        "audit_manifest_sha256": config.audit_manifest_sha256,
        "tokenizer_version_manifest_sha256": (config.tokenizer_version_manifest_sha256),
        "tokenizer_sha256": config.tokenizer_sha256,
        "token_dtype": config.token_dtype,
        "encode_special_tokens_as_text": True,
    }
    if config.input_split != "train":
        identity["input_split"] = config.input_split
    return identity


def _validate_lineage(
    root: Path, config: FormalTokenShardConfig
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_manifest_path = root / config.split_dir / "manifest.json"
    paths = (
        (split_manifest_path, config.split_manifest_sha256, "split manifest"),
        (root / config.audit_manifest, config.audit_manifest_sha256, "audit manifest"),
        (
            root / config.tokenizer_version_manifest,
            config.tokenizer_version_manifest_sha256,
            "tokenizer version manifest",
        ),
        (root / config.tokenizer_path, config.tokenizer_sha256, "tokenizer"),
    )
    for path, expected, label in paths:
        if not path.is_file():
            raise FormalTokenShardError(f"{label} is missing: {path}")
        if file_sha256(path) != expected:
            raise FormalTokenShardError(f"{label} SHA-256 mismatch")
    split = _load_json(split_manifest_path, "split manifest")
    audit = _load_json(root / config.audit_manifest, "audit manifest")
    tokenizer_version = _load_json(
        root / config.tokenizer_version_manifest, "tokenizer version manifest"
    )
    if audit.get("training_eligible") is not True:
        raise FormalTokenShardError("formal data audit is not training eligible")
    if audit.get("provenance", {}).get("split") != config.split_manifest_sha256:
        raise FormalTokenShardError("audit does not bind the configured split")
    if tokenizer_version.get("formal_pretraining_eligible") is not True:
        raise FormalTokenShardError("tokenizer version is not pretraining eligible")
    tokenizer_lineage = tokenizer_version.get("lineage", {}).get("tokenizer", {})
    if tokenizer_lineage.get("tokenizer_sha256") != config.tokenizer_sha256:
        raise FormalTokenShardError("tokenizer version does not bind tokenizer file")
    shards = split.get("shards", {}).get(config.input_split)
    if not isinstance(shards, list) or not shards:
        raise FormalTokenShardError(
            f"split manifest has no {config.input_split} shards"
        )
    return split, audit, tokenizer_version


def _verify_completed_shard(directory: Path, item: dict[str, Any]) -> None:
    document_count = item.get("document_count")
    token_count = item.get("token_count")
    if type(document_count) is not int or document_count <= 0:
        raise FormalTokenShardError("completed shard document_count is invalid")
    if type(token_count) is not int or token_count <= 0:
        raise FormalTokenShardError("completed shard token_count is invalid")
    for key in ("token_file", "index_file"):
        metadata = item.get(key)
        if not isinstance(metadata, dict):
            raise FormalTokenShardError(f"completed shard {key} metadata is invalid")
        path = directory / metadata.get("name", "")
        if not path.is_file() or path.stat().st_size != metadata.get("size_bytes"):
            raise FormalTokenShardError(
                f"completed shard {key} is missing or truncated"
            )
        if file_sha256(path) != metadata.get("sha256"):
            raise FormalTokenShardError(f"completed shard {key} SHA-256 mismatch")
    if item["token_file"].get("dtype") != "uint16-le":
        raise FormalTokenShardError("completed shard token dtype is invalid")
    if item["token_file"]["size_bytes"] != token_count * 2:
        raise FormalTokenShardError("completed shard token size is inconsistent")
    if item["index_file"].get("dtype") != "uint64-le":
        raise FormalTokenShardError("completed shard index dtype is invalid")
    if item["index_file"].get("shape") != [document_count, 2]:
        raise FormalTokenShardError("completed shard index shape is inconsistent")
    if item["index_file"]["size_bytes"] != document_count * 2 * 8:
        raise FormalTokenShardError("completed shard index size is inconsistent")


def verify_formal_token_shards(directory: str | Path) -> dict[str, Any]:
    """Verify one completed formal token-shard artifact and all shard hashes."""
    output_dir = Path(directory)
    manifest_path = output_dir / MANIFEST_NAME
    completed_path = output_dir / COMPLETED_NAME
    if not manifest_path.is_file() or not completed_path.is_file():
        raise FormalTokenShardError("formal token-shard artifact is incomplete")
    manifest = _load_json(manifest_path, "token-shard manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise FormalTokenShardError("token-shard schema version is unsupported")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise FormalTokenShardError("token-shard format version is unsupported")
    if manifest.get("token_dtype") != "uint16-le":
        raise FormalTokenShardError("token-shard dtype is unsupported")
    if manifest.get("encode_special_tokens_as_text") is not True:
        raise FormalTokenShardError("token-shard special-token encoding is unsafe")
    if manifest.get("formal_training_eligible") is not True:
        raise FormalTokenShardError("token-shard artifact is not training eligible")
    expected_marker = f"{file_sha256(manifest_path)}  {MANIFEST_NAME}\n"
    if completed_path.read_text(encoding="utf-8") != expected_marker:
        raise FormalTokenShardError("token-shard COMPLETED marker is invalid")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise FormalTokenShardError("token-shard manifest has no shards")
    for item in shards:
        _verify_completed_shard(output_dir, item)
    if manifest.get("document_count") != sum(item["document_count"] for item in shards):
        raise FormalTokenShardError("token-shard document total is inconsistent")
    if manifest.get("token_count") != sum(item["token_count"] for item in shards):
        raise FormalTokenShardError("token-shard token total is inconsistent")
    return manifest


def _encode_one_shard(
    *,
    source_path: Path,
    source: dict[str, Any],
    output_dir: Path,
    output_index: int,
    tokenizer: Tokenizer,
    max_rss_bytes: int,
    progress_interval: int,
) -> dict[str, Any]:
    if file_sha256(source_path) != source.get("sha256"):
        raise FormalTokenShardError(f"input shard SHA-256 mismatch: {source_path.name}")
    base = f"part-{output_index:05d}"
    token_final = output_dir / f"{base}.bin"
    index_final = output_dir / f"{base}.idx"
    token_tmp = output_dir / f".{base}.bin.tmp"
    index_tmp = output_dir / f".{base}.idx.tmp"
    token_tmp.unlink(missing_ok=True)
    index_tmp.unlink(missing_ok=True)
    document_count = 0
    token_count = 0
    unknown_count = 0
    started = time.monotonic()
    last_progress = started
    try:
        with (
            source_path.open(encoding="utf-8") as source_handle,
            token_tmp.open("wb") as token_handle,
            index_tmp.open("wb") as index_handle,
        ):
            for line in source_handle:
                document = CanonicalDocument.from_json_line(line)
                ids = tokenizer.encode(document.text, add_special_tokens=False).ids
                unknown_count += ids.count(1)
                document_tokens = np.empty(len(ids) + 2, dtype="<u2")
                document_tokens[0] = 2
                document_tokens[-1] = 3
                document_tokens[1:-1] = ids
                document_tokens.tofile(token_handle)
                np.asarray((token_count, len(document_tokens)), dtype="<u8").tofile(
                    index_handle
                )
                token_count += len(document_tokens)
                document_count += 1
                current_rss = _rss_bytes()
                if current_rss > max_rss_bytes:
                    raise FormalTokenShardError(
                        f"peak RSS exceeded configured limit: {current_rss / _GIB:.2f}GiB"
                    )
                now = time.monotonic()
                if now - last_progress >= progress_interval:
                    print(
                        f"[token-shards] source={source_path.name} "
                        f"documents={document_count} tokens={token_count} "
                        f"rss={current_rss / _GIB:.2f}GiB",
                        flush=True,
                    )
                    last_progress = now
            for handle in (token_handle, index_handle):
                handle.flush()
                os.fsync(handle.fileno())
        if document_count != source.get("record_count"):
            raise FormalTokenShardError(
                f"input record count mismatch for {source_path.name}: "
                f"expected {source.get('record_count')}, got {document_count}"
            )
        if unknown_count:
            raise FormalTokenShardError(
                f"tokenizer emitted {unknown_count} unexpected unknown tokens"
            )
        token_tmp.replace(token_final)
        index_tmp.replace(index_final)
        return {
            "source_name": source_path.name,
            "source_sha256": source["sha256"],
            "document_count": document_count,
            "token_count": token_count,
            "token_file": {
                "name": token_final.name,
                "dtype": "uint16-le",
                "size_bytes": token_final.stat().st_size,
                "sha256": file_sha256(token_final),
            },
            "index_file": {
                "name": index_final.name,
                "dtype": "uint64-le",
                "shape": [document_count, 2],
                "size_bytes": index_final.stat().st_size,
                "sha256": file_sha256(index_final),
            },
        }
    except BaseException:
        token_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
        raise


def _encode_shard_worker(
    *,
    source_path: str,
    source: dict[str, Any],
    output_dir: str,
    output_index: int,
    tokenizer_path: str,
    max_rss_bytes: int,
    progress_interval: int,
) -> tuple[int, dict[str, Any]]:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.encode_special_tokens = True
    item = _encode_one_shard(
        source_path=Path(source_path),
        source=source,
        output_dir=Path(output_dir),
        output_index=output_index,
        tokenizer=tokenizer,
        max_rss_bytes=max_rss_bytes,
        progress_interval=progress_interval,
    )
    return output_index, item


def build_formal_token_shards(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
    max_input_shards: int | None = None,
) -> dict[str, Any]:
    """Build or resume the configured formal token-shard artifact."""
    root = Path(project_root).resolve()
    config = load_formal_token_shard_config(root / config_path)
    split, audit, tokenizer_version = _validate_lineage(root, config)
    identity = _identity(config)
    identity_sha = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    completed_path = output_dir / COMPLETED_NAME
    if manifest_path.exists() or completed_path.exists():
        if not manifest_path.is_file() or not completed_path.is_file():
            raise FormalTokenShardError("formal token-shard artifact is incomplete")
        manifest = verify_formal_token_shards(output_dir)
        if manifest.get("identity") != identity:
            raise FormalTokenShardError("existing token-shard identity mismatch")
        return manifest

    state_path = output_dir / STATE_NAME
    if state_path.exists():
        state = _load_json(state_path, "token-shard state")
        if state.get("identity") != identity:
            raise FormalTokenShardError("resume state identity mismatch")
    else:
        state = {"schema_version": SCHEMA_VERSION, "identity": identity, "shards": []}
        _write_json_atomic(state_path, state)
    completed_shards = state.get("shards")
    if not isinstance(completed_shards, list):
        raise FormalTokenShardError("resume state shards must be a list")
    for item in completed_shards:
        _verify_completed_shard(output_dir, item)

    source_shards = split["shards"][config.input_split]
    if len(completed_shards) > len(source_shards):
        raise FormalTokenShardError("resume state contains too many shards")
    for index, item in enumerate(completed_shards):
        source = source_shards[index]
        if (
            item.get("source_name") != source.get("name")
            or item.get("source_sha256") != source.get("sha256")
            or item.get("document_count") != source.get("record_count")
        ):
            raise FormalTokenShardError(
                f"resume state source lineage mismatch at shard {index}"
            )
    limit = len(source_shards)
    if max_input_shards is not None:
        if max_input_shards <= 0:
            raise ValueError("max_input_shards must be positive")
        limit = min(limit, max_input_shards)
    tokenizer = Tokenizer.from_file(str(root / config.tokenizer_path))
    tokenizer.encode_special_tokens = True
    if tokenizer.get_vocab_size(with_added_tokens=True) > np.iinfo(np.uint16).max + 1:
        raise FormalTokenShardError("tokenizer vocabulary does not fit uint16")
    for token, expected_id in (("<unk>", 1), ("<bos>", 2), ("<eos>", 3)):
        if tokenizer.token_to_id(token) != expected_id:
            raise FormalTokenShardError(f"tokenizer {token} ID mismatch")

    start_index = len(completed_shards)
    for batch_start in range(start_index, limit, config.workers):
        batch_indices = list(
            range(batch_start, min(batch_start + config.workers, limit))
        )
        for index in batch_indices:
            print(
                f"[token-shards] start {index + 1}/{len(source_shards)} "
                f"source={source_shards[index]['name']}",
                flush=True,
            )
        if config.workers == 1:
            results = []
            for index in batch_indices:
                source = source_shards[index]
                source_path = (
                    root
                    / config.split_dir
                    / config.input_split
                    / "shards"
                    / source["name"]
                )
                item = _encode_one_shard(
                    source_path=source_path,
                    source=source,
                    output_dir=output_dir,
                    output_index=index,
                    tokenizer=tokenizer,
                    max_rss_bytes=int(config.max_rss_gib * _GIB),
                    progress_interval=config.progress_interval_seconds,
                )
                results.append((index, item))
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=config.workers
            ) as executor:
                futures = []
                for index in batch_indices:
                    source = source_shards[index]
                    source_path = (
                        root
                        / config.split_dir
                        / config.input_split
                        / "shards"
                        / source["name"]
                    )
                    futures.append(
                        executor.submit(
                            _encode_shard_worker,
                            source_path=str(source_path),
                            source=source,
                            output_dir=str(output_dir),
                            output_index=index,
                            tokenizer_path=str(root / config.tokenizer_path),
                            max_rss_bytes=int(config.max_rss_gib * _GIB),
                            progress_interval=config.progress_interval_seconds,
                        )
                    )
                results = [future.result() for future in futures]
        for index, item in sorted(results):
            completed_shards.append(item)
            print(
                f"[token-shards] finish {index + 1}/{len(source_shards)} "
                f"documents={item['document_count']} tokens={item['token_count']}",
                flush=True,
            )
        _write_json_atomic(state_path, state)

    if limit < len(source_shards):
        return state
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "dataset_id": f"token-shards-{config.name}-{identity_sha[:12]}",
        "identity_sha256": identity_sha,
        "identity": identity,
        "data_version": {
            "split_manifest_sha256": config.split_manifest_sha256,
            "audit_manifest_sha256": config.audit_manifest_sha256,
            "training_eligible": audit["training_eligible"],
        },
        "tokenizer": {
            "version_id": tokenizer_version["tokenizer_version_id"],
            "tokenizer_sha256": config.tokenizer_sha256,
            "vocab_size": tokenizer_version["contract"]["vocab_size"],
        },
        "token_dtype": config.token_dtype,
        "split": config.input_split,
        "encode_special_tokens_as_text": True,
        "workers": config.workers,
        "index_columns": ["token_offset", "token_count"],
        "document_count": sum(item["document_count"] for item in completed_shards),
        "token_count": sum(item["token_count"] for item in completed_shards),
        "peak_rss_limit_bytes": int(config.max_rss_gib * _GIB),
        "observed_peak_rss_bytes": _rss_bytes(),
        "shards": completed_shards,
        "formal_training_eligible": True,
    }
    _write_json_atomic(manifest_path, manifest)
    completed_path.write_text(
        f"{file_sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
    )
    state_path.unlink()
    return manifest


def inspect_formal_token_shard_inputs(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate formal lineage and report the work without writing output."""
    root = Path(project_root).resolve()
    config = load_formal_token_shard_config(root / config_path)
    split, audit, tokenizer_version = _validate_lineage(root, config)
    shards = split["shards"][config.input_split]
    return {
        "config": str(config_path),
        "input_shards": len(shards),
        "input_split": config.input_split,
        "input_documents": sum(item["record_count"] for item in shards),
        "input_jsonl_bytes": sum(
            (root / config.split_dir / config.input_split / "shards" / item["name"])
            .stat()
            .st_size
            for item in shards
        ),
        "token_dtype": config.token_dtype,
        "bytes_per_token": 2,
        "max_rss_gib": config.max_rss_gib,
        "workers": config.workers,
        "data_training_eligible": audit["training_eligible"],
        "tokenizer_version_id": tokenizer_version["tokenizer_version_id"],
        "output_dir": str(config.output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate lineage and report input size without writing shards",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            _canonical_json(
                inspect_formal_token_shard_inputs(
                    args.config,
                    project_root=args.project_root,
                )
            )
        )
        return 0
    manifest = build_formal_token_shards(
        args.config,
        project_root=args.project_root,
    )
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
