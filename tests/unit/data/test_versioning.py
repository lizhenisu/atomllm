import hashlib
import json
from pathlib import Path

import pytest

from atomllm.data.versioning import DataVersionError, build_data_version


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def build_synthetic_lineage(root: Path) -> tuple[Path, Path, Path, Path]:
    acquisition = root / "acquisition"
    cleaning = root / "cleaning"
    deduplication = root / "deduplication"
    splitting = root / "splitting"
    for directory in (acquisition, cleaning, deduplication, splitting):
        directory.mkdir()

    acquisition_documents = acquisition / "documents.jsonl"
    acquisition_documents.write_text('{"stage":"acquisition"}\n', encoding="utf-8")
    acquisition_manifest = {
        "purpose": "acquisition_smoke",
        "source_enabled": False,
        "source_id": "synthetic-version-v1",
        "dataset": "example-invalid/dataset",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "config_name": "synthetic.zh",
        "split": "train",
        "record_count": 1,
        "documents_sha256": sha256(acquisition_documents),
    }
    write_json(acquisition / "manifest.json", acquisition_manifest)

    cleaning_documents = cleaning / "documents.jsonl"
    cleaning_documents.write_text('{"stage":"cleaning"}\n', encoding="utf-8")
    cleaning_manifest = {
        "input_documents_sha256": sha256(acquisition_documents),
        "input_manifest_sha256": sha256(acquisition / "manifest.json"),
        "record_count": 1,
        "dropped_count": 0,
        "documents_sha256": sha256(cleaning_documents),
        "transform": {
            "privacy_action": "warn",
            "quality_action": "warn",
        },
    }
    write_json(cleaning / "manifest.json", cleaning_manifest)

    clusters = deduplication / "duplicate-clusters.jsonl"
    clusters.write_text("", encoding="utf-8")
    deduplication_manifest = {
        "clusters_file": clusters.name,
        "clusters_sha256": sha256(clusters),
        "input_documents_sha256": sha256(cleaning_documents),
        "input_manifest_sha256": sha256(cleaning / "manifest.json"),
        "record_count": 1,
        "action": "report_only",
        "dropped_count": 0,
        "exact_cluster_count": 0,
        "near_cluster_count": 0,
    }
    write_json(deduplication / "manifest.json", deduplication_manifest)

    split_contents = {
        "assignments": '{"document_id":"doc-synthetic","split":"train"}\n',
        "train": '{"stage":"cleaning"}\n',
        "validation": "",
        "test": "",
    }
    split_files = {}
    for name, content in split_contents.items():
        path = splitting / f"{name}.jsonl"
        path.write_text(content, encoding="utf-8")
        split_files[name] = {
            "name": path.name,
            "record_count": len(content.splitlines()),
            "sha256": sha256(path),
        }
    splitting_manifest = {
        "input_documents_sha256": sha256(cleaning_documents),
        "input_manifest_sha256": sha256(cleaning / "manifest.json"),
        "deduplication_manifest_sha256": sha256(deduplication / "manifest.json"),
        "duplicate_clusters_sha256": sha256(clusters),
        "frozen": True,
        "record_count": 1,
        "overlap_document_count": 0,
        "cross_split_duplicate_cluster_count": 0,
        "split_counts": {"train": 1, "validation": 0, "test": 0},
        "files": split_files,
    }
    write_json(splitting / "manifest.json", splitting_manifest)
    return acquisition, cleaning, deduplication, splitting


def test_build_data_version_verifies_complete_lineage(tmp_path: Path) -> None:
    stages = build_synthetic_lineage(tmp_path)
    output = tmp_path / "output"

    manifest = build_data_version(*stages, output)

    assert manifest["status"] == "smoke_validated"
    assert manifest["training_eligible"] is False
    assert manifest["lineage"]["splitting"]["split_counts"] == {
        "train": 1,
        "validation": 0,
        "test": 0,
    }
    assert manifest["audit"]["all_manifest_links_verified"] is True
    checksum = (output / "manifest.sha256").read_text(encoding="utf-8")
    assert checksum.endswith("  manifest.json\n")


def test_data_version_is_byte_for_byte_idempotent(tmp_path: Path) -> None:
    stages = build_synthetic_lineage(tmp_path)
    output = tmp_path / "output"

    first = build_data_version(*stages, output)
    first_bytes = (output / "manifest.json").read_bytes()
    second = build_data_version(*stages, output)

    assert second == first
    assert (output / "manifest.json").read_bytes() == first_bytes


def test_modified_acquisition_document_is_rejected(tmp_path: Path) -> None:
    stages = build_synthetic_lineage(tmp_path)
    with (stages[0] / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(DataVersionError, match="acquisition documents SHA-256"):
        build_data_version(*stages, tmp_path / "output")


def test_broken_manifest_link_is_rejected(tmp_path: Path) -> None:
    stages = build_synthetic_lineage(tmp_path)
    cleaning_manifest_path = stages[1] / "manifest.json"
    cleaning_manifest = json.loads(cleaning_manifest_path.read_text(encoding="utf-8"))
    cleaning_manifest["input_manifest_sha256"] = "0" * 64
    write_json(cleaning_manifest_path, cleaning_manifest)

    with pytest.raises(DataVersionError, match="cleaning input manifest"):
        build_data_version(*stages, tmp_path / "output")


def test_modified_existing_version_checksum_is_rejected(tmp_path: Path) -> None:
    stages = build_synthetic_lineage(tmp_path)
    output = tmp_path / "output"
    build_data_version(*stages, output)
    (output / "manifest.sha256").write_text("invalid\n", encoding="utf-8")

    with pytest.raises(DataVersionError, match="checksum is invalid"):
        build_data_version(*stages, output)
