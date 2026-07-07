"""Deterministically encode canonical documents into fixed-length token blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from atomllm.data.schema import CanonicalDocument
from atomllm.model.config import CORE_SPECIAL_TOKEN_IDS, load_model_config
from atomllm.training.config import file_sha256, load_training_config


PACKED_DATA_SCHEMA_VERSION = 1
PACKING_VERSION = "document-bos-eos-fixed-block-v1"
TOKEN_FILE_NAME = "tokens.bin"
MANIFEST_FILE_NAME = "manifest.json"
COMPLETED_FILE_NAME = "COMPLETED"


class PackingError(RuntimeError):
    """Raised when token packing cannot be completed or verified safely."""


@dataclass(frozen=True, slots=True)
class PackingSpec:
    name: str
    data_version_id: str
    input_path: Path
    input_sha256: str
    tokenizer_version_id: str
    tokenizer_path: Path
    tokenizer_sha256: str
    vocab_size: int
    sequence_length: int
    output_dir: Path


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackingError(f"cannot read packed-data manifest: {path}") from error
    if not isinstance(value, dict):
        raise PackingError("packed-data manifest must be an object")
    return value


def _packing_identity(spec: PackingSpec) -> dict[str, Any]:
    return {
        "data_version_id": spec.data_version_id,
        "input_sha256": spec.input_sha256,
        "name": spec.name,
        "packing_version": PACKING_VERSION,
        "sequence_length": spec.sequence_length,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "tokenizer_version_id": spec.tokenizer_version_id,
        "vocab_size": spec.vocab_size,
    }


def _validate_spec(spec: PackingSpec) -> None:
    if spec.sequence_length < 2:
        raise PackingError("sequence_length must be at least 2")
    if spec.vocab_size <= max(CORE_SPECIAL_TOKEN_IDS.values()):
        raise PackingError("vocab_size is too small for the special-token protocol")
    if not spec.input_path.is_file():
        raise FileNotFoundError(f"training split not found: {spec.input_path}")
    if not spec.tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {spec.tokenizer_path}")
    if file_sha256(spec.input_path) != spec.input_sha256:
        raise PackingError("training split SHA-256 does not match")
    if file_sha256(spec.tokenizer_path) != spec.tokenizer_sha256:
        raise PackingError("tokenizer SHA-256 does not match")


def verify_packed_dataset(
    output_dir: str | Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify completion marker, identity, file size, and SHA-256."""
    directory = Path(output_dir)
    manifest_path = directory / MANIFEST_FILE_NAME
    completed_path = directory / COMPLETED_FILE_NAME
    token_path = directory / TOKEN_FILE_NAME
    if not directory.is_dir():
        raise PackingError(f"packed-data directory not found: {directory}")
    if not manifest_path.is_file() or not completed_path.is_file():
        raise PackingError("packed-data artifact is incomplete")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != PACKED_DATA_SCHEMA_VERSION:
        raise PackingError("packed-data schema version is unsupported")
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        raise PackingError("packed-data identity does not match packing spec")
    manifest_sha256 = file_sha256(manifest_path)
    if completed_path.read_text(encoding="utf-8") != (
        f"{manifest_sha256}  {MANIFEST_FILE_NAME}\n"
    ):
        raise PackingError("packed-data COMPLETED marker is invalid")
    token_file = manifest.get("token_file")
    if not isinstance(token_file, dict):
        raise PackingError("packed-data token_file metadata is invalid")
    if token_file.get("name") != TOKEN_FILE_NAME or not token_path.is_file():
        raise PackingError("packed-data token file is missing")
    if token_file.get("dtype") != "uint32-le":
        raise PackingError("packed-data token dtype is unsupported")
    size_bytes = token_file.get("size_bytes")
    sha256 = token_file.get("sha256")
    if type(size_bytes) is not int or size_bytes < 0:
        raise PackingError("packed-data token size is invalid")
    if token_path.stat().st_size != size_bytes:
        raise PackingError("packed-data token file size does not match")
    if file_sha256(token_path) != sha256:
        raise PackingError("packed-data token SHA-256 does not match")
    block_count = manifest.get("block_count")
    sequence_length = manifest.get("sequence_length")
    if (
        type(block_count) is not int
        or block_count <= 0
        or type(sequence_length) is not int
        or sequence_length < 2
        or size_bytes != block_count * sequence_length * 4
    ):
        raise PackingError("packed-data token shape is inconsistent")
    return manifest


def pack_token_dataset(spec: PackingSpec) -> dict[str, Any]:
    """Create or idempotently verify one deterministic packed-token artifact."""
    _validate_spec(spec)
    identity = _packing_identity(spec)
    if spec.output_dir.exists():
        return verify_packed_dataset(
            spec.output_dir,
            expected_identity=identity,
        )

    tokenizer = Tokenizer.from_file(str(spec.tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != spec.vocab_size:
        raise PackingError("tokenizer vocabulary size does not match")
    for name, token, token_id in (
        ("bos", "<bos>", CORE_SPECIAL_TOKEN_IDS["bos"]),
        ("eos", "<eos>", CORE_SPECIAL_TOKEN_IDS["eos"]),
        ("unk", "<unk>", CORE_SPECIAL_TOKEN_IDS["unk"]),
    ):
        if tokenizer.token_to_id(token) != token_id:
            raise PackingError(f"tokenizer {name} token ID does not match")

    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{spec.output_dir.name}.tmp-",
            dir=spec.output_dir.parent,
        )
    )
    token_path = temporary_dir / TOKEN_FILE_NAME
    document_count = 0
    encoded_text_tokens = 0
    boundary_tokens = 0
    written_tokens = 0
    unknown_tokens = 0
    carry: list[int] = []
    try:
        with (
            spec.input_path.open(encoding="utf-8") as input_handle,
            token_path.open("wb") as output_handle,
        ):
            for line in input_handle:
                document = CanonicalDocument.from_json_line(line)
                encoding = tokenizer.encode(
                    document.text,
                    add_special_tokens=False,
                )
                unknown_tokens += encoding.ids.count(CORE_SPECIAL_TOKEN_IDS["unk"])
                document_count += 1
                encoded_text_tokens += len(encoding.ids)
                boundary_tokens += 2
                combined = [
                    *carry,
                    CORE_SPECIAL_TOKEN_IDS["bos"],
                    *encoding.ids,
                    CORE_SPECIAL_TOKEN_IDS["eos"],
                ]
                full_token_count = (
                    len(combined) // spec.sequence_length * spec.sequence_length
                )
                if full_token_count:
                    np.asarray(
                        combined[:full_token_count],
                        dtype="<u4",
                    ).tofile(output_handle)
                    written_tokens += full_token_count
                carry = combined[full_token_count:]
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if document_count == 0 or written_tokens == 0:
            raise PackingError("training split produced no complete token blocks")
        if unknown_tokens:
            raise PackingError("tokenizer produced unexpected unknown tokens")

        block_count = written_tokens // spec.sequence_length
        token_sha256 = file_sha256(token_path)
        identity_sha256 = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": PACKED_DATA_SCHEMA_VERSION,
            "packed_data_id": f"packed-{spec.name}-{identity_sha256[:12]}",
            "identity_sha256": identity_sha256,
            "identity": identity,
            "document_count": document_count,
            "encoded_text_tokens": encoded_text_tokens,
            "boundary_tokens": boundary_tokens,
            "total_source_tokens": encoded_text_tokens + boundary_tokens,
            "packed_tokens": written_tokens,
            "dropped_tail_tokens": len(carry),
            "block_count": block_count,
            "sequence_length": spec.sequence_length,
            "token_file": {
                "name": TOKEN_FILE_NAME,
                "dtype": "uint32-le",
                "shape": [block_count, spec.sequence_length],
                "size_bytes": token_path.stat().st_size,
                "sha256": token_sha256,
            },
            "formal_training_eligible": False,
        }
        manifest_path = temporary_dir / MANIFEST_FILE_NAME
        manifest_path.write_text(
            f"{json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)}\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        completed_path = temporary_dir / COMPLETED_FILE_NAME
        completed_path.write_text(
            f"{file_sha256(manifest_path)}  {MANIFEST_FILE_NAME}\n",
            encoding="utf-8",
        )
        with completed_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_dir, spec.output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return verify_packed_dataset(spec.output_dir, expected_identity=identity)


def build_spec(
    *,
    training_config_path: Path,
    input_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    project_root: Path,
) -> PackingSpec:
    config = load_training_config(
        training_config_path,
        project_root=project_root,
    )
    model_config = load_model_config(project_root / config.model.config_path)
    return PackingSpec(
        name=f"{config.name}-seq{config.batch.sequence_length}",
        data_version_id=config.data.data_version_id,
        input_path=input_path,
        input_sha256=config.data.split_sha256,
        tokenizer_version_id=config.data.tokenizer_version_id,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=config.data.tokenizer_sha256,
        vocab_size=model_config.tokenizer.vocab_size,
        sequence_length=config.batch.sequence_length,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack the bound training split into deterministic token blocks."
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/atom-5m-baseline.yaml"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-split-v1/train.jsonl"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/atom-tokenizer-smoke-v1/tokenizer.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training-data/atom-5m-wikipedia-128-v1"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    spec = build_spec(
        training_config_path=args.training_config,
        input_path=(project_root / args.input).resolve(),
        tokenizer_path=(project_root / args.tokenizer).resolve(),
        output_dir=(project_root / args.output_dir).resolve(),
        project_root=project_root,
    )
    manifest = pack_token_dataset(spec)
    summary = {
        key: manifest[key]
        for key in (
            "packed_data_id",
            "document_count",
            "packed_tokens",
            "dropped_tail_tokens",
            "block_count",
            "sequence_length",
            "formal_training_eligible",
        )
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
