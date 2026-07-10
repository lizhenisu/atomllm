import hashlib
import json
from pathlib import Path

from atomllm.data.formal_cleaning import clean_formal
from atomllm.data.formal_fingerprint import fingerprint_formal
from atomllm.data.formal_exact_dedup import exact_deduplicate_formal
from atomllm.data.formal_near_dedup import (
    PROBE_COUNT,
    _probe_masks,
    near_deduplicate_formal,
)
from atomllm.data.formal_split_config import load_formal_split_config
from atomllm.data.schema import CanonicalDocument, make_document_id


def _document(index: int, text: str) -> CanonicalDocument:
    source_id = "synthetic-formal-70g"
    record_id = f"record-{index}"
    return CanonicalDocument.from_mapping(
        {
            "schema_version": 1,
            "document_id": make_document_id(source_id, record_id),
            "source_id": source_id,
            "source_record_id": record_id,
            "text": text,
            "language": "zh-Hans",
            "content_type": "general",
            "privacy_warnings": [],
            "quality_warnings": [],
            "metadata": {},
        }
    )


def _write_input(path: Path, documents: list[CanonicalDocument]) -> None:
    path.mkdir()
    payload = "".join(f"{document.to_json_line()}\n" for document in documents)
    documents_path = path / "documents.jsonl"
    documents_path.write_text(payload, encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "record_count": len(documents),
                "documents_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_stream_cleaning_writes_fixed_shards_and_is_idempotent(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_input(
        input_dir,
        [_document(index, f"第{index}篇合成文本。  \n" * 30) for index in range(5)],
    )

    manifest = clean_formal(
        input_dir=input_dir,
        output_dir=output_dir,
        records_per_shard=2,
    )
    repeated = clean_formal(
        input_dir=input_dir,
        output_dir=output_dir,
        records_per_shard=2,
    )

    assert repeated == manifest
    assert manifest["record_count"] == 5
    assert manifest["shard_count"] == 3
    assert [item["record_count"] for item in manifest["shards"]] == [2, 2, 1]
    assert (output_dir / "state.json").read_text(encoding="utf-8").find(
        '"completed": true'
    ) >= 0


def test_fingerprint_shards_match_cleaned_shards(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    clean_dir = tmp_path / "clean"
    fingerprint_dir = tmp_path / "fingerprints"
    _write_input(
        input_dir,
        [_document(index, f"第{index}篇合成文本。" * 30) for index in range(3)],
    )
    clean_formal(input_dir=input_dir, output_dir=clean_dir, records_per_shard=2)

    manifest = fingerprint_formal(clean_dir=clean_dir, output_dir=fingerprint_dir)
    repeated = fingerprint_formal(clean_dir=clean_dir, output_dir=fingerprint_dir)

    assert repeated == manifest
    assert manifest["record_count"] == 3
    values = [
        json.loads(line)
        for shard in manifest["shards"]
        for line in (fingerprint_dir / "shards" / shard["name"])
        .read_text()
        .splitlines()
    ]
    assert {value["document_id"] for value in values} == {
        _document(index, f"第{index}篇合成文本。" * 30).document_id
        for index in range(3)
    }
    assert all(len(value["text_sha256"]) == 64 for value in values)
    assert all(len(value["simhash"]) == 16 for value in values)


def test_external_exact_dedup_uses_disk_index(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    clean_dir = tmp_path / "clean"
    fingerprint_dir = tmp_path / "fingerprints"
    exact_dir = tmp_path / "exact"
    _write_input(
        input_dir,
        [
            _document(0, "重复文本。" * 30),
            _document(1, "重复文本。" * 30),
            _document(2, "不同文本。" * 30),
        ],
    )
    clean_formal(input_dir=input_dir, output_dir=clean_dir, records_per_shard=2)
    fingerprint_formal(clean_dir=clean_dir, output_dir=fingerprint_dir)

    manifest = exact_deduplicate_formal(
        fingerprint_dir=fingerprint_dir,
        output_dir=exact_dir,
    )
    repeated = exact_deduplicate_formal(
        fingerprint_dir=fingerprint_dir,
        output_dir=exact_dir,
    )

    assert repeated == manifest
    assert manifest["record_count"] == 3
    assert manifest["representative_count"] == 2
    assert manifest["exact_duplicate_document_count"] == 1


def test_external_near_dedup_writes_cluster_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    clean_dir = tmp_path / "clean"
    fingerprint_dir = tmp_path / "fingerprints"
    exact_dir = tmp_path / "exact"
    near_dir = tmp_path / "near"
    _write_input(
        input_dir,
        [_document(index, f"第{index}篇不同的合成文本。" * 30) for index in range(3)],
    )
    clean_formal(input_dir=input_dir, output_dir=clean_dir, records_per_shard=2)
    fingerprint_formal(clean_dir=clean_dir, output_dir=fingerprint_dir)
    exact_deduplicate_formal(fingerprint_dir=fingerprint_dir, output_dir=exact_dir)

    manifest = near_deduplicate_formal(
        exact_dir=exact_dir,
        clean_dir=clean_dir,
        fingerprint_dir=fingerprint_dir,
        output_dir=near_dir,
    )
    repeated = near_deduplicate_formal(
        exact_dir=exact_dir,
        clean_dir=clean_dir,
        fingerprint_dir=fingerprint_dir,
        output_dir=near_dir,
    )

    assert repeated == manifest
    assert (near_dir / manifest["clusters_file"]).is_file()
    assert manifest["near_duplicate_pair_count"] >= 0
    assert manifest["lsh"]["probe_count"] == PROBE_COUNT


def test_near_dedup_uses_stable_32_bit_external_probes() -> None:
    first = _probe_masks()

    assert first == _probe_masks()
    assert len(first) == PROBE_COUNT == len(set(first))
    assert all(mask.bit_count() == 32 for mask in first)


def test_default_formal_split_contract_is_train_validation_only() -> None:
    config = load_formal_split_config("configs/data/formal-70g-processing.yaml")

    assert (config.train_fraction, config.validation_fraction) == (0.99, 0.01)
    assert (config.min_estimated_tokens, config.max_estimated_tokens) == (512, 7168)
    assert config.validation_quality_warnings == "ascending"
    assert config.stable_tie_breaker == "sha256"
