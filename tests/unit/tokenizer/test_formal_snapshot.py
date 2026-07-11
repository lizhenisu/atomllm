import hashlib
import json
from pathlib import Path

import pytest

from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.tokenizer.config import load_tokenizer_config
from atomllm.tokenizer.formal_snapshot import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TOKENIZER_OUTPUT_DIR,
    FormalTokenizerSnapshotError,
    build_formal_tokenizer_snapshot,
    build_parser,
    load_formal_snapshot_config,
)


CONFIG_PATH = Path("configs/tokenizer/formal-snapshot-v1.yaml")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document(index: int, *, language: str, content_type: str) -> CanonicalDocument:
    source_id = "synthetic-source"
    source_record_id = f"record-{index}"
    return CanonicalDocument(
        schema_version=1,
        document_id=make_document_id(source_id, source_record_id),
        source_id=source_id,
        source_record_id=source_record_id,
        text=f"synthetic document {index} {language} {content_type}",
        language=language,
        content_type=content_type,
        privacy_warnings=(),
        quality_warnings=(),
        metadata={},
    )


def write_inputs(root: Path) -> tuple[Path, Path]:
    split_dir = root / "split"
    shard_dir = split_dir / "train" / "shards"
    shard_dir.mkdir(parents=True)
    documents = [
        document(0, language="zh-Hans", content_type="general"),
        document(1, language="en", content_type="code"),
        document(2, language="zh-Hant", content_type="encyclopedia"),
        document(3, language="ja", content_type="general"),
        document(4, language="fr", content_type="math"),
    ]
    shard_path = shard_dir / "part-00000.jsonl"
    shard_path.write_text(
        "".join(f"{item.to_json_line()}\n" for item in documents), encoding="utf-8"
    )
    split_manifest = {
        "training_eligible": True,
        "shards": {
            "train": [
                {
                    "name": shard_path.name,
                    "record_count": len(documents),
                    "sha256": sha256(shard_path),
                }
            ]
        },
    }
    split_manifest_path = split_dir / "manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")
    audit_dir = root / "audit"
    audit_dir.mkdir()
    audit_manifest = {
        "training_eligible": True,
        "checks": {"all": True},
        "provenance": {"split": sha256(split_manifest_path)},
    }
    (audit_dir / "manifest.json").write_text(
        json.dumps(audit_manifest), encoding="utf-8"
    )
    return split_dir, audit_dir


def test_loads_committed_snapshot_config() -> None:
    config = load_formal_snapshot_config(CONFIG_PATH)

    assert config.sample_ratio == 0.01
    assert config.split_dir == DEFAULT_SPLIT_DIR
    assert config.audit_dir == DEFAULT_AUDIT_DIR
    assert config.output_dir == DEFAULT_OUTPUT_DIR
    assert config.tokenizer_output_dir == DEFAULT_TOKENIZER_OUTPUT_DIR
    assert config.progress_interval_seconds == 60.0


def test_snapshot_cli_only_exposes_config() -> None:
    parser = build_parser()

    assert {action.dest for action in parser._actions} == {"help", "config"}


def test_snapshot_config_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(FormalTokenizerSnapshotError, match="missing required fields"):
        load_formal_snapshot_config(path)


def test_snapshot_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.yaml"
    path.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")

    with pytest.raises(FormalTokenizerSnapshotError, match="unknown fields"):
        load_formal_snapshot_config(path)


def test_builds_idempotent_snapshot_and_release_training_config(tmp_path: Path) -> None:
    split_dir, audit_dir = write_inputs(tmp_path)
    output_dir = tmp_path / "snapshot"
    tokenizer_dir = tmp_path / "tokenizer"

    first = build_formal_tokenizer_snapshot(
        project_root=tmp_path,
        split_dir=split_dir,
        audit_dir=audit_dir,
        output_dir=output_dir,
        tokenizer_output_dir=tokenizer_dir,
        sample_ratio=1.0,
        progress_interval_seconds=60.0,
    )
    second = build_formal_tokenizer_snapshot(
        project_root=tmp_path,
        split_dir=split_dir,
        audit_dir=audit_dir,
        output_dir=output_dir,
        tokenizer_output_dir=tokenizer_dir,
        sample_ratio=1.0,
        progress_interval_seconds=60.0,
    )

    assert second == first
    assert first["snapshot"]["document_count"] == 5
    assert first["snapshot"]["document_ratio"] == 1.0
    assert first["data_version_id"].startswith("data-formal-70g-tokenizer-snapshot-v1-")
    config = load_tokenizer_config(output_dir / "tokenizer-training.yaml")
    assert config.status == "release"
    assert config.training_eligible is True
    assert "digits" in config.evaluation.suites


def test_rejects_invalid_sample_ratio(tmp_path: Path) -> None:
    split_dir, audit_dir = write_inputs(tmp_path)

    with pytest.raises(FormalTokenizerSnapshotError, match="sample_ratio"):
        build_formal_tokenizer_snapshot(
            project_root=tmp_path,
            split_dir=split_dir,
            audit_dir=audit_dir,
            output_dir=tmp_path / "snapshot",
            tokenizer_output_dir=tmp_path / "tokenizer",
            sample_ratio=0.0,
            progress_interval_seconds=60.0,
        )


def test_rejects_audit_for_a_different_split(tmp_path: Path) -> None:
    split_dir, audit_dir = write_inputs(tmp_path)
    audit_path = audit_dir / "manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["provenance"]["split"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(FormalTokenizerSnapshotError, match="does not match split"):
        build_formal_tokenizer_snapshot(
            project_root=tmp_path,
            split_dir=split_dir,
            audit_dir=audit_dir,
            output_dir=tmp_path / "snapshot",
            tokenizer_output_dir=tmp_path / "tokenizer",
            sample_ratio=1.0,
            progress_interval_seconds=60.0,
        )
