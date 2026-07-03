import hashlib
import json
from pathlib import Path

import pytest

from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.data.splitting import SplittingError, split_dataset


def make_document(index: int) -> CanonicalDocument:
    source_id = "synthetic-split-v1"
    record_id = f"record-{index}"
    return CanonicalDocument.from_mapping(
        {
            "schema_version": 1,
            "document_id": make_document_id(source_id, record_id),
            "source_id": source_id,
            "source_record_id": record_id,
            "text": f"第{index}篇互不相同的合成文章。" * 20,
            "language": "zh-Hans" if index % 2 == 0 else "zh-Hant",
            "content_type": "encyclopedia",
            "privacy_warnings": [],
            "quality_warnings": [],
            "metadata": {"fixture": True},
        }
    )


def write_input_dataset(path: Path, documents: list[CanonicalDocument]) -> str:
    path.mkdir()
    documents_path = path / "documents.jsonl"
    documents_path.write_text(
        "".join(f"{document.to_json_line()}\n" for document in documents),
        encoding="utf-8",
    )
    digest = hashlib.sha256(documents_path.read_bytes()).hexdigest()
    manifest = {
        "source_id": "synthetic-split-v1",
        "record_count": len(documents),
        "documents_sha256": digest,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return digest


def write_deduplication_result(
    path: Path,
    input_digest: str,
    record_count: int,
    clusters: list[dict],
) -> None:
    path.mkdir()
    clusters_path = path / "duplicate-clusters.jsonl"
    clusters_path.write_text(
        "".join(
            f"{json.dumps(cluster, sort_keys=True, separators=(',', ':'))}\n"
            for cluster in clusters
        ),
        encoding="utf-8",
    )
    manifest = {
        "input_documents_sha256": input_digest,
        "record_count": record_count,
        "clusters_sha256": hashlib.sha256(clusters_path.read_bytes()).hexdigest(),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def read_assignments(path: Path) -> dict[str, str]:
    return {
        value["document_id"]: value["split"]
        for value in (
            json.loads(line)
            for line in (path / "assignments.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }


def test_split_creates_exact_smoke_quotas_and_disjoint_files(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    deduplication_dir = tmp_path / "deduplication"
    output_dir = tmp_path / "output"
    documents = [make_document(index) for index in range(100)]
    digest = write_input_dataset(input_dir, documents)
    write_deduplication_result(deduplication_dir, digest, 100, [])

    manifest = split_dataset(input_dir, deduplication_dir, output_dir)
    assignments = read_assignments(output_dir)

    assert manifest["split_counts"] == {
        "train": 98,
        "validation": 1,
        "test": 1,
    }
    assert len(assignments) == 100
    split_ids = {
        split: {
            json.loads(line)["document_id"]
            for line in (output_dir / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        for split in ("train", "validation", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert set().union(*split_ids.values()) == set(assignments)


def test_duplicate_cluster_never_crosses_splits(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    deduplication_dir = tmp_path / "deduplication"
    output_dir = tmp_path / "output"
    documents = [make_document(index) for index in range(100)]
    digest = write_input_dataset(input_dir, documents)
    cluster = {
        "cluster_id": "cluster-synthetic",
        "kind": "near",
        "representative_document_id": documents[0].document_id,
        "members": [documents[0].document_id, documents[1].document_id],
    }
    write_deduplication_result(deduplication_dir, digest, 100, [cluster])

    manifest = split_dataset(input_dir, deduplication_dir, output_dir)
    assignments = read_assignments(output_dir)

    assert (
        assignments[documents[0].document_id] == assignments[documents[1].document_id]
    )
    assert manifest["cross_split_duplicate_cluster_count"] == 0


def test_split_is_idempotent(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    deduplication_dir = tmp_path / "deduplication"
    output_dir = tmp_path / "output"
    documents = [make_document(index) for index in range(20)]
    digest = write_input_dataset(input_dir, documents)
    write_deduplication_result(deduplication_dir, digest, 20, [])

    first = split_dataset(input_dir, deduplication_dir, output_dir)
    second = split_dataset(input_dir, deduplication_dir, output_dir)

    assert second == first


def test_tampered_split_file_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    deduplication_dir = tmp_path / "deduplication"
    output_dir = tmp_path / "output"
    documents = [make_document(index) for index in range(20)]
    digest = write_input_dataset(input_dir, documents)
    write_deduplication_result(deduplication_dir, digest, 20, [])
    split_dataset(input_dir, deduplication_dir, output_dir)
    with (output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(SplittingError, match="SHA-256 mismatch"):
        split_dataset(input_dir, deduplication_dir, output_dir)


def test_unknown_duplicate_member_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    deduplication_dir = tmp_path / "deduplication"
    output_dir = tmp_path / "output"
    documents = [make_document(index) for index in range(20)]
    digest = write_input_dataset(input_dir, documents)
    cluster = {
        "cluster_id": "cluster-invalid",
        "kind": "exact",
        "representative_document_id": documents[0].document_id,
        "members": [documents[0].document_id, "doc-" + "0" * 64],
    }
    write_deduplication_result(deduplication_dir, digest, 20, [cluster])

    with pytest.raises(SplittingError, match="unknown document"):
        split_dataset(input_dir, deduplication_dir, output_dir)
