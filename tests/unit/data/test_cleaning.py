import hashlib
import json
from pathlib import Path

import pytest

import atomllm.data.cleaning as cleaning
from atomllm.data.cleaning import (
    CleaningError,
    clean_dataset,
    clean_document,
    normalize_text,
    quality_warnings,
)
from atomllm.data.schema import CanonicalDocument, make_document_id


def make_document(
    index: int,
    text: str,
    *,
    language: str = "zh-Hans",
) -> CanonicalDocument:
    source_id = "synthetic-clean-v1"
    record_id = f"record-{index}"
    return CanonicalDocument.from_mapping(
        {
            "schema_version": 1,
            "document_id": make_document_id(source_id, record_id),
            "source_id": source_id,
            "source_record_id": record_id,
            "text": text,
            "language": language,
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
        "source_id": "synthetic-clean-v1",
        "record_count": len(documents),
        "documents_sha256": digest,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def test_normalize_text_is_deterministic_and_idempotent() -> None:
    original = " Cafe\u0301  \r\n\r\n\r\n第二行\t \r\n"

    normalized, changes = normalize_text(original)
    repeated, repeated_changes = normalize_text(normalized)

    assert normalized == "Café\n\n第二行"
    assert changes == (
        "unicode_nfc",
        "line_endings",
        "trailing_whitespace",
        "excess_blank_lines",
        "outer_whitespace",
    )
    assert repeated == normalized
    assert repeated_changes == ()


def test_quality_warnings_are_conservative_labels() -> None:
    warnings = quality_warnings(
        f"{'甲' * 20}\ufffd",
        "zh",
    )

    assert warnings == (
        "too_short",
        "decode_replacement",
        "high_repetition",
        "low_language_confidence",
    )


def test_clean_document_preserves_identity_and_privacy_warnings() -> None:
    original = make_document(1, "合成内容  \n")
    original = CanonicalDocument.from_mapping(
        {
            **original.to_mapping(),
            "privacy_warnings": ["email"],
            "quality_warnings": ["suspected_boilerplate"],
        }
    )

    result = clean_document(original)

    assert result.document.document_id == original.document_id
    assert result.document.text == "合成内容"
    assert result.document.privacy_warnings == ("email",)
    assert result.document.quality_warnings == (
        "suspected_boilerplate",
        "too_short",
    )


def test_clean_dataset_writes_integrity_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_input_dataset(
        input_dir,
        [
            make_document(1, "第一篇合成文本。" * 30),
            make_document(2, "第二篇合成文本。  \n" * 30),
        ],
    )

    manifest = clean_dataset(input_dir, output_dir)

    assert manifest["record_count"] == 2
    assert manifest["retained_count"] == 2
    assert manifest["dropped_count"] == 0
    assert manifest["changed_document_count"] == 1
    assert manifest["change_counts"] == {
        "outer_whitespace": 1,
        "trailing_whitespace": 1,
    }
    output = output_dir / "documents.jsonl"
    assert (
        hashlib.sha256(output.read_bytes()).hexdigest() == manifest["documents_sha256"]
    )


def test_clean_dataset_resumes_after_transform_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    documents = [
        make_document(index, f"第{index}篇合成文本。" * 30) for index in range(3)
    ]
    write_input_dataset(input_dir, documents)
    original_clean_document = cleaning.clean_document
    calls = 0

    def interrupted_clean(document: CanonicalDocument):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic interruption")
        return original_clean_document(document)

    monkeypatch.setattr(cleaning, "clean_document", interrupted_clean)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        clean_dataset(input_dir, output_dir)

    state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
    assert state["records_written"] == 2
    monkeypatch.setattr(cleaning, "clean_document", original_clean_document)

    manifest = clean_dataset(input_dir, output_dir)

    assert manifest["record_count"] == 3
    assert len((output_dir / "documents.jsonl").read_text().splitlines()) == 3


def test_clean_dataset_rejects_modified_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_input_dataset(input_dir, [make_document(1, "合成文本。" * 30)])
    with (input_dir / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(CleaningError, match="SHA-256"):
        clean_dataset(input_dir, output_dir)
