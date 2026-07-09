import hashlib
import json
from pathlib import Path

import pytest

from atomllm.data.formal_processing import (
    FormalProcessingError,
    process_formal_v0,
)
from atomllm.data.schema import CanonicalDocument, make_document_id


def make_document(index: int, text: str) -> CanonicalDocument:
    source_id = "synthetic-formal-v0"
    source_record_id = f"record-{index}"
    return CanonicalDocument.from_mapping(
        {
            "schema_version": 1,
            "document_id": make_document_id(source_id, source_record_id),
            "source_id": source_id,
            "source_record_id": source_record_id,
            "text": text,
            "language": "zh-Hans" if index % 2 == 0 else "en",
            "content_type": "general",
            "privacy_warnings": ["email"] if index == 0 else [],
            "quality_warnings": [],
            "metadata": {"estimated_tokens": len(text), "fixture": True},
        }
    )


def write_acquired_dataset(
    path: Path,
    documents: list[CanonicalDocument],
    *,
    training_eligible: bool = True,
) -> None:
    path.mkdir(parents=True)
    documents_path = path / "documents.jsonl"
    documents_path.write_text(
        "".join(f"{document.to_json_line()}\n" for document in documents),
        encoding="utf-8",
    )
    manifest = {
        "formal_training_eligible": training_eligible,
        "record_count": len(documents),
        "estimated_tokens": sum(
            int(document.metadata["estimated_tokens"]) for document in documents
        ),
        "documents_sha256": hashlib.sha256(documents_path.read_bytes()).hexdigest(),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def test_process_formal_v0_runs_clean_dedup_and_split(tmp_path: Path) -> None:
    acquired_dir = tmp_path / "acquired"
    clean_dir = tmp_path / "clean"
    deduplication_dir = tmp_path / "dedup"
    split_dir = tmp_path / "split"
    processing_dir = tmp_path / "processed"
    documents = [
        make_document(index, f"第{index}篇正式v0合成文档。" * 30)
        for index in range(100)
    ]
    write_acquired_dataset(acquired_dir, documents)

    manifest = process_formal_v0(
        acquired_dir=acquired_dir,
        clean_dir=clean_dir,
        deduplication_dir=deduplication_dir,
        split_dir=split_dir,
        processing_dir=processing_dir,
    )

    assert manifest["formal_training_eligible"] is True
    assert manifest["cleaning"]["record_count"] == 100
    assert manifest["deduplication"]["record_count"] == 100
    assert manifest["splitting"]["split_counts"] == {
        "train": 98,
        "validation": 1,
        "test": 1,
    }
    assert (processing_dir / "manifest.json").is_file()


def test_process_formal_v0_rejects_unapproved_acquisition(tmp_path: Path) -> None:
    acquired_dir = tmp_path / "acquired"
    write_acquired_dataset(
        acquired_dir,
        [make_document(0, "未放行的合成文档。" * 30)],
        training_eligible=False,
    )

    with pytest.raises(FormalProcessingError, match="not training eligible"):
        process_formal_v0(acquired_dir=acquired_dir, processing_dir=tmp_path / "out")
