from pathlib import Path

import numpy as np
import pytest

from atomllm.training.config import file_sha256
from atomllm.training.packing import (
    PackingError,
    PackingSpec,
    TOKEN_FILE_NAME,
    pack_token_dataset,
    verify_packed_dataset,
)


def test_packing_is_deterministic_and_preserves_document_boundaries(
    packed_dataset_dir: Path,
) -> None:
    manifest = verify_packed_dataset(packed_dataset_dir)
    tokens = np.memmap(
        packed_dataset_dir / TOKEN_FILE_NAME,
        mode="r",
        dtype="<u4",
        shape=(manifest["block_count"], manifest["sequence_length"]),
    )

    assert manifest["document_count"] == 10
    assert manifest["block_count"] == 10
    assert manifest["dropped_tail_tokens"] == 0
    assert tokens[0].tolist() == [2, 4, 14, 3]
    assert tokens[1].tolist() == [2, 5, 14, 3]


def test_existing_packed_artifact_is_verified_idempotently(
    packed_dataset_dir: Path,
) -> None:
    manifest = verify_packed_dataset(packed_dataset_dir)
    identity = manifest["identity"]
    input_path = packed_dataset_dir.parent / "train.jsonl"
    tokenizer_path = packed_dataset_dir.parent / "tokenizer.json"
    spec = PackingSpec(
        name=identity["name"],
        data_version_id=identity["data_version_id"],
        input_path=input_path,
        input_sha256=file_sha256(input_path),
        tokenizer_version_id=identity["tokenizer_version_id"],
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=file_sha256(tokenizer_path),
        vocab_size=identity["vocab_size"],
        sequence_length=identity["sequence_length"],
        output_dir=packed_dataset_dir,
    )

    second = pack_token_dataset(spec)
    assert second == manifest


def test_packed_artifact_rejects_token_file_tampering(
    packed_dataset_dir: Path,
) -> None:
    token_path = packed_dataset_dir / TOKEN_FILE_NAME
    original = token_path.read_bytes()
    token_path.write_bytes(b"\xff" + original[1:])

    with pytest.raises(PackingError, match="SHA-256"):
        verify_packed_dataset(packed_dataset_dir)


def test_packing_rejects_input_hash_drift(packed_dataset_dir: Path) -> None:
    manifest = verify_packed_dataset(packed_dataset_dir)
    identity = manifest["identity"]
    input_path = packed_dataset_dir.parent / "train.jsonl"
    tokenizer_path = packed_dataset_dir.parent / "tokenizer.json"
    spec = PackingSpec(
        name="different-output",
        data_version_id=identity["data_version_id"],
        input_path=input_path,
        input_sha256="0" * 64,
        tokenizer_version_id=identity["tokenizer_version_id"],
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=file_sha256(tokenizer_path),
        vocab_size=identity["vocab_size"],
        sequence_length=identity["sequence_length"],
        output_dir=packed_dataset_dir.parent / "different",
    )

    with pytest.raises(PackingError, match="training split SHA-256"):
        pack_token_dataset(spec)
