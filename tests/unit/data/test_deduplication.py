import hashlib
import json
from pathlib import Path

import pytest

from atomllm.data.deduplication import (
    DeduplicationError,
    analyze_duplicates,
)
from atomllm.data.schema import CanonicalDocument, make_document_id


def make_document(index: int, text: str) -> CanonicalDocument:
    source_id = "synthetic-dedup-v1"
    record_id = f"record-{index}"
    return CanonicalDocument.from_mapping(
        {
            "schema_version": 1,
            "document_id": make_document_id(source_id, record_id),
            "source_id": source_id,
            "source_record_id": record_id,
            "text": text,
            "language": "zh-Hans",
            "content_type": "encyclopedia",
            "privacy_warnings": [],
            "quality_warnings": [],
            "metadata": {"fixture": True},
        }
    )


def write_input_dataset(path: Path, documents: list[CanonicalDocument]) -> None:
    path.mkdir()
    documents_path = path / "documents.jsonl"
    documents_path.write_text(
        "".join(f"{document.to_json_line()}\n" for document in documents),
        encoding="utf-8",
    )
    digest = hashlib.sha256(documents_path.read_bytes()).hexdigest()
    manifest = {
        "source_id": "synthetic-dedup-v1",
        "record_count": len(documents),
        "documents_sha256": digest,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def read_clusters(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (path / "duplicate-clusters.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_exact_duplicates_form_a_cluster_without_deletion(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    duplicate_text = "这是一篇完全相同的合成文章。" * 30
    documents = [
        make_document(0, duplicate_text),
        make_document(1, duplicate_text),
        make_document(2, "这是一篇不同的合成文章。" * 30),
    ]
    write_input_dataset(input_dir, documents)

    manifest = analyze_duplicates(input_dir, output_dir)
    clusters = read_clusters(output_dir)

    assert manifest["exact_cluster_count"] == 1
    assert manifest["exact_duplicate_document_count"] == 1
    assert manifest["retained_count"] == 3
    assert manifest["dropped_count"] == 0
    assert clusters[0]["kind"] == "exact"
    assert clusters[0]["representative_document_id"] == documents[0].document_id
    assert clusters[0]["members"] == [
        documents[0].document_id,
        documents[1].document_id,
    ]


def test_near_duplicates_are_reported_conservatively(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    base = "".join(f"第{index}段包含唯一的合成百科说明文字。" for index in range(100))
    near = base.replace("第50段包含", "第50段经过修改并包含")
    documents = [
        make_document(0, base),
        make_document(1, near),
        make_document(2, "完全无关的短篇合成内容。" * 100),
    ]
    write_input_dataset(input_dir, documents)

    manifest = analyze_duplicates(input_dir, output_dir)
    near_clusters = [
        cluster for cluster in read_clusters(output_dir) if cluster["kind"] == "near"
    ]

    assert manifest["near_cluster_count"] == 1
    assert manifest["near_duplicate_pair_count"] == 1
    assert near_clusters[0]["members"] == [
        documents[0].document_id,
        documents[1].document_id,
    ]
    assert near_clusters[0]["similarity_edges"][0]["similarity"] >= 0.90


def test_completed_analysis_is_idempotent(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_input_dataset(
        input_dir,
        [make_document(0, "第一篇合成内容。" * 30)],
    )

    first = analyze_duplicates(input_dir, output_dir)
    second = analyze_duplicates(input_dir, output_dir)

    assert second == first


def test_modified_cluster_report_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_input_dataset(
        input_dir,
        [make_document(0, "第一篇合成内容。" * 30)],
    )
    analyze_duplicates(input_dir, output_dir)
    (output_dir / "duplicate-clusters.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DeduplicationError, match="SHA-256"):
        analyze_duplicates(input_dir, output_dir)


def test_modified_input_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_input_dataset(
        input_dir,
        [make_document(0, "第一篇合成内容。" * 30)],
    )
    with (input_dir / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(DeduplicationError, match="SHA-256"):
        analyze_duplicates(input_dir, output_dir)
