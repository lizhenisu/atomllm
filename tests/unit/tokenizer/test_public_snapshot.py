import json
from pathlib import Path

import pytest

from atomllm.tokenizer.public_snapshot import (
    PublicTokenizerSnapshotError,
    _heldout_score,
    _offer_heldout,
    _sha256,
    _selected,
    _training_config,
    _verify_audit,
    build,
)
from atomllm.data.schema import CanonicalDocument, make_document_id


def test_snapshot_selection_is_deterministic_and_nested() -> None:
    document_ids = [f"doc-{index:064x}" for index in range(1000)]
    half = {item for item in document_ids if _selected(item, 0.5, 42)}
    larger = {item for item in document_ids if _selected(item, 0.7, 42)}

    assert half
    assert half < larger
    assert 450 <= len(half) <= 550
    assert 650 <= len(larger) <= 750


def test_snapshot_rejects_unsafe_artifact_label(tmp_path: Path) -> None:
    with pytest.raises(PublicTokenizerSnapshotError, match="artifact_label"):
        build(
            source_dir=Path("missing-source"),
            audit_dir=Path("missing-audit"),
            output_dir=Path("missing-output"),
            tokenizer_output_dir=Path("missing-tokenizers"),
            sample_ratio=0.84,
            artifact_label="084pct/unsafe",
            project_root=tmp_path,
        )


def test_heldout_keeps_lowest_hashes_from_each_source() -> None:
    heap = []
    offered = []
    for index in range(20):
        record_id = f"row-{index}"
        document = CanonicalDocument(
            schema_version=1,
            document_id=make_document_id("public-source", record_id),
            source_id="public-source",
            source_record_id=record_id,
            text=f"real public document {index}",
            language="en",
            content_type="general",
            privacy_warnings=(),
            quality_warnings=(),
            metadata={},
        )
        _offer_heldout(
            heap,
            document=document,
            line=document.to_json_line() + "\n",
            text_bytes=len(document.text.encode()),
            limit=5,
            seed=42,
        )
        score = _heldout_score(document.document_id, 42)
        offered.append((score, document.document_id))

    assert len(heap) == 5
    assert {item[1] for item in heap} == {
        document_id for _score, document_id in sorted(offered)[:5]
    }


def test_heldout_reservation_precedes_training_sampling() -> None:
    document_id = next(
        item
        for index in range(1000)
        if _selected((item := f"doc-{index:064x}"), 0.9, 42)
    )
    document = CanonicalDocument(
        schema_version=1,
        document_id=document_id,
        source_id="public-source",
        source_record_id="selected-training-row",
        text="real public document selected by the training hash",
        language="en",
        content_type="general",
        privacy_warnings=(),
        quality_warnings=(),
        metadata={},
    )
    heap = []

    _offer_heldout(
        heap,
        document=document,
        line=document.to_json_line() + "\n",
        text_bytes=len(document.text.encode()),
        limit=100,
        seed=43,
    )

    assert [entry[1] for entry in heap] == [document_id]


def test_heldout_selection_is_independent_of_source_order() -> None:
    documents = [
        CanonicalDocument(
            schema_version=1,
            document_id=make_document_id("public-source", f"row-{index}"),
            source_id="public-source",
            source_record_id=f"row-{index}",
            text=f"real public document {index}",
            language="en",
            content_type="general",
            privacy_warnings=(),
            quality_warnings=(),
            metadata={},
        )
        for index in range(200)
    ]

    def selected(items: list[CanonicalDocument]) -> set[str]:
        heap = []
        for document in items:
            _offer_heldout(
                heap,
                document=document,
                line=document.to_json_line() + "\n",
                text_bytes=len(document.text.encode()),
                limit=100,
                seed=43,
            )
        return {entry[1] for entry in heap}

    assert selected(documents) == selected(list(reversed(documents)))


def test_snapshot_requires_audit_bound_to_exact_corpus(tmp_path) -> None:
    source = tmp_path / "corpus"
    audit = tmp_path / "audit"
    source.mkdir()
    audit.mkdir()
    (source / "manifest.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    (source / "documents.jsonl").write_text('{"text": "example"}\n', encoding="utf-8")
    report = {
        "training_eligible": True,
        "corpus_manifest_sha256": _sha256(source / "manifest.json"),
        "documents_sha256": _sha256(source / "documents.jsonl"),
    }
    report_path = audit / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (audit / "COMPLETED").write_text(
        f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
    )

    assert _verify_audit(audit, source)["training_eligible"] is True

    (source / "documents.jsonl").write_text('{"text": "changed"}\n', encoding="utf-8")
    with pytest.raises(PublicTokenizerSnapshotError, match="document mismatch"):
        _verify_audit(audit, source)


def test_candidate_training_config_uses_requested_vocab_and_output() -> None:
    config = _training_config(
        name="atom-tokenizer-en-zh-050pct-48k-v1",
        vocab_size=48000,
        data_version_id="data-public-example",
        document_count=123,
        documents_sha256="a" * 64,
        documents_path=Path("artifacts/snapshot/documents.jsonl"),
        tokenizer_output_dir=Path("artifacts/tokenizers/candidate-48k"),
    )

    assert config["algorithm"]["vocab_size"] == 48000
    assert config["output_dir"] == "artifacts/tokenizers/candidate-48k"
