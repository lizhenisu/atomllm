import hashlib
import json
from pathlib import Path

from atomllm.data.formal_audit import audit_formal_v0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_documents(directory: Path, count: int) -> str:
    documents = directory / "documents.jsonl"
    documents.write_text(
        "".join(f'{{"document_id":"doc-{index:03d}"}}\n' for index in range(count)),
        encoding="utf-8",
    )
    return sha256(documents)


def build_lineage(root: Path, *, source_limit_passes: bool) -> dict[str, Path]:
    acquired = root / "acquired"
    clean = root / "clean"
    dedup = root / "dedup"
    split = root / "split"
    processed = root / "processed"
    for directory in (acquired, clean, dedup, split, processed):
        directory.mkdir()

    acquired_sha = write_documents(acquired, 100)
    source_tokens = (
        {
            "source-a": 190_000,
            "source-b": 190_000,
            "source-c": 190_000,
            "source-d": 190_000,
            "source-e": 180_000,
            "source-f": 60_000,
        }
        if source_limit_passes
        else {"source-a": 500_000, "source-b": 500_000}
    )
    write_json(
        acquired / "manifest.json",
        {
            "documents_file": "documents.jsonl",
            "documents_sha256": acquired_sha,
            "estimated_tokens": 1_000_000,
            "estimated_tokens_by_content": {
                "code": 160_000,
                "encyclopedia": 220_000,
                "general": 420_000,
                "math": 100_000,
                "science": 100_000,
            },
            "estimated_tokens_by_language_bucket": {
                "en": 300_000,
                "ja": 70_000,
                "other": 30_000,
                "zh-Hans": 500_000,
                "zh-Hant": 100_000,
            },
            "estimated_tokens_by_source": source_tokens,
            "formal_training_eligible": True,
            "privacy_action": "warn",
            "record_count": 100,
            "uncovered_content_buckets": [],
            "uncovered_language_buckets": [],
        },
    )

    clean_sha = write_documents(clean, 100)
    write_json(
        clean / "manifest.json",
        {
            "documents_file": "documents.jsonl",
            "documents_sha256": clean_sha,
            "dropped_count": 0,
            "input_documents_sha256": acquired_sha,
            "input_manifest_sha256": sha256(acquired / "manifest.json"),
            "privacy_warning_counts": {},
            "quality_warning_counts": {},
            "record_count": 100,
            "transform": {"privacy_action": "warn", "quality_action": "warn"},
        },
    )

    clusters = dedup / "duplicate-clusters.jsonl"
    clusters.write_text("", encoding="utf-8")
    write_json(
        dedup / "manifest.json",
        {
            "action": "report_only",
            "clusters_file": "duplicate-clusters.jsonl",
            "clusters_sha256": sha256(clusters),
            "dropped_count": 0,
            "exact_cluster_count": 0,
            "exact_duplicate_document_count": 0,
            "input_documents_sha256": clean_sha,
            "input_manifest_sha256": sha256(clean / "manifest.json"),
            "near_candidate_document_count": 0,
            "near_cluster_count": 0,
            "record_count": 100,
        },
    )

    split_files = {
        "assignments": '{"document_id":"doc-000","split":"train"}\n' * 100,
        "train": '{"document_id":"doc"}\n' * 98,
        "validation": '{"document_id":"doc"}\n',
        "test": '{"document_id":"doc"}\n',
    }
    files = {}
    for name, content in split_files.items():
        path = split / f"{name}.jsonl"
        path.write_text(content, encoding="utf-8")
        files[name] = {
            "name": path.name,
            "record_count": len(content.splitlines()),
            "sha256": sha256(path),
        }
    write_json(
        split / "manifest.json",
        {
            "cross_split_duplicate_cluster_count": 0,
            "deduplication_manifest_sha256": sha256(dedup / "manifest.json"),
            "duplicate_clusters_sha256": sha256(clusters),
            "files": files,
            "frozen": True,
            "input_documents_sha256": clean_sha,
            "input_manifest_sha256": sha256(clean / "manifest.json"),
            "overlap_document_count": 0,
            "record_count": 100,
            "split_counts": {"train": 98, "validation": 1, "test": 1},
        },
    )
    write_json(
        processed / "manifest.json",
        {"formal_training_eligible": True},
    )
    return {
        "acquired": acquired,
        "clean": clean,
        "dedup": dedup,
        "split": split,
        "processed": processed,
    }


def test_formal_audit_passes_complete_lineage(tmp_path: Path) -> None:
    paths = build_lineage(tmp_path, source_limit_passes=True)

    manifest = audit_formal_v0(
        acquired_dir=paths["acquired"],
        clean_dir=paths["clean"],
        deduplication_dir=paths["dedup"],
        split_dir=paths["split"],
        processing_dir=paths["processed"],
        audit_dir=tmp_path / "audit",
    )

    assert manifest["status"] == "passed"
    assert manifest["formal_training_eligible"] is True
    assert manifest["failure_count"] == 0


def test_formal_audit_blocks_source_fraction_violation(tmp_path: Path) -> None:
    paths = build_lineage(tmp_path, source_limit_passes=False)

    manifest = audit_formal_v0(
        acquired_dir=paths["acquired"],
        clean_dir=paths["clean"],
        deduplication_dir=paths["dedup"],
        split_dir=paths["split"],
        processing_dir=paths["processed"],
        audit_dir=tmp_path / "audit",
    )

    assert manifest["status"] == "blocked"
    assert manifest["formal_training_eligible"] is False
    assert [failure["name"] for failure in manifest["failures"]] == [
        "source_fraction_limit"
    ]
