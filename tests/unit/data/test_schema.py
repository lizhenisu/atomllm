import json
from pathlib import Path

import pytest

from atomllm.data.schema import (
    CanonicalDocument,
    DataSchemaError,
    SourceRegistry,
    load_source_registry,
    make_document_id,
)


VALID_SOURCE = {
    "source_id": "synthetic-smoke-v1",
    "name": "Synthetic Smoke Corpus",
    "version": "v1",
    "license": "CC0-1.0",
    "homepage": "https://example.invalid/synthetic-smoke",
    "languages": ["zh", "en"],
    "content_types": ["general"],
    "data_format": "jsonl",
    "enabled": False,
    "acquisition": {
        "provider": "synthetic",
        "location": "synthetic://smoke-corpus",
        "revision": "v1",
        "expected_sha256": None,
    },
}


def valid_registry(*sources: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "privacy_action": "warn",
        "sources": list(sources),
    }


def valid_document() -> dict[str, object]:
    source_id = "synthetic-smoke-v1"
    source_record_id = "record-000001"
    return {
        "schema_version": 1,
        "document_id": make_document_id(source_id, source_record_id),
        "source_id": source_id,
        "source_record_id": source_record_id,
        "text": "合成示例文本。",
        "language": "zh",
        "content_type": "general",
        "privacy_warnings": [],
        "quality_warnings": [],
        "metadata": {"synthetic": True},
    }


def test_loads_default_empty_source_registry() -> None:
    registry = load_source_registry("configs/data/sources.yaml")

    assert registry == SourceRegistry(
        schema_version=1,
        privacy_action="warn",
        sources=(),
    )


def test_parses_and_serializes_valid_source() -> None:
    registry = SourceRegistry.from_mapping(valid_registry(VALID_SOURCE))

    assert registry.sources[0].source_id == "synthetic-smoke-v1"
    assert registry.sources[0].languages == ("zh", "en")
    assert registry.to_mapping() == valid_registry(VALID_SOURCE)


def test_rejects_duplicate_source_ids() -> None:
    with pytest.raises(DataSchemaError, match="source_id values must be unique"):
        SourceRegistry.from_mapping(valid_registry(VALID_SOURCE, VALID_SOURCE))


def test_rejects_unknown_source_field() -> None:
    source = dict(VALID_SOURCE)
    source["unexpected"] = "synthetic"

    with pytest.raises(DataSchemaError, match="unknown fields: unexpected"):
        SourceRegistry.from_mapping(valid_registry(source))


def test_requires_warning_only_privacy_policy() -> None:
    registry = valid_registry()
    registry["privacy_action"] = "reject"

    with pytest.raises(DataSchemaError, match="privacy_action must be 'warn'"):
        SourceRegistry.from_mapping(registry)


@pytest.mark.parametrize("revision", ["latest", "main", "HEAD"])
def test_rejects_floating_source_revision(revision: str) -> None:
    source = json.loads(json.dumps(VALID_SOURCE))
    source["acquisition"]["revision"] = revision

    with pytest.raises(DataSchemaError, match="revision must be immutable"):
        SourceRegistry.from_mapping(valid_registry(source))


def test_rejects_floating_source_version() -> None:
    source = dict(VALID_SOURCE)
    source["version"] = "latest"

    with pytest.raises(DataSchemaError, match="version must be immutable"):
        SourceRegistry.from_mapping(valid_registry(source))


def test_rejects_credentials_in_homepage() -> None:
    source = dict(VALID_SOURCE)
    source["homepage"] = "https://synthetic:secret@example.invalid/corpus"

    with pytest.raises(DataSchemaError, match="homepage must not contain credentials"):
        SourceRegistry.from_mapping(valid_registry(source))


def test_rejects_unsafe_local_path() -> None:
    source = json.loads(json.dumps(VALID_SOURCE))
    source["acquisition"] = {
        "provider": "local",
        "location": "../private/raw.jsonl",
        "revision": "v1",
        "expected_sha256": "a" * 64,
    }

    with pytest.raises(DataSchemaError, match="safe relative path"):
        SourceRegistry.from_mapping(valid_registry(source))


def test_document_id_is_stable_and_source_scoped() -> None:
    first = make_document_id("synthetic-a-v1", "record-1")
    repeated = make_document_id("synthetic-a-v1", "record-1")
    other_source = make_document_id("synthetic-b-v1", "record-1")

    assert first == repeated
    assert first != other_source
    assert first.startswith("doc-")
    assert len(first) == 68


def test_document_jsonl_round_trip_preserves_unicode() -> None:
    document = CanonicalDocument.from_mapping(valid_document())

    restored = CanonicalDocument.from_json_line(document.to_json_line())

    assert restored == document
    assert restored.text == "合成示例文本。"


def test_privacy_warning_is_accepted_not_rejected() -> None:
    raw_document = valid_document()
    raw_document["privacy_warnings"] = ["email", "phone"]

    document = CanonicalDocument.from_mapping(raw_document)

    assert document.privacy_warnings == ("email", "phone")


def test_rejects_unknown_document_field() -> None:
    raw_document = valid_document()
    raw_document["unexpected"] = "synthetic"

    with pytest.raises(DataSchemaError, match="unknown fields: unexpected"):
        CanonicalDocument.from_mapping(raw_document)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_rejects_empty_document_text(text: str) -> None:
    raw_document = valid_document()
    raw_document["text"] = text

    with pytest.raises(DataSchemaError, match="text must be a non-empty string"):
        CanonicalDocument.from_mapping(raw_document)


def test_rejects_document_id_that_does_not_match_source() -> None:
    raw_document = valid_document()
    raw_document["document_id"] = f"doc-{'0' * 64}"

    with pytest.raises(DataSchemaError, match="does not match source identity"):
        CanonicalDocument.from_mapping(raw_document)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), object()])
def test_rejects_non_json_metadata_value(invalid_value: object) -> None:
    raw_document = valid_document()
    raw_document["metadata"] = {"invalid": invalid_value}

    with pytest.raises(DataSchemaError, match="metadata"):
        CanonicalDocument.from_mapping(raw_document)


def test_rejects_malformed_document_json() -> None:
    with pytest.raises(DataSchemaError, match="invalid document JSON"):
        CanonicalDocument.from_json_line('{"schema_version":')


def test_missing_source_registry_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source registry not found"):
        load_source_registry(tmp_path / "missing.yaml")
