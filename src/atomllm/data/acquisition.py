"""Resumable acquisition utilities for the first Wikipedia data smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import opencc as opencc_package
from dotenv import load_dotenv
from opencc import OpenCC

from atomllm.data.schema import (
    SCHEMA_VERSION,
    CanonicalDocument,
    DataSchemaError,
    DataSource,
    load_source_registry,
    make_document_id,
)


_TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")
_SIMPLIFIED_TO_TRADITIONAL = OpenCC("s2t")


def chinese_script_classifier_identity() -> dict[str, str]:
    """Fingerprint the locked classification rules without transforming corpus text."""
    package_root = Path(opencc_package.__file__).resolve().parent
    paths = [
        package_root / "config/s2t.json",
        package_root / "config/t2s.json",
        *sorted((package_root / "dictionary").glob("*.txt")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"OpenCC classification rule is missing: {path.name}")
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "backend": "opencc-python-reimplemented",
        "distribution_version": importlib.metadata.version(
            "opencc-python-reimplemented"
        ),
        "rules_sha256": digest.hexdigest(),
    }


class AcquisitionError(RuntimeError):
    """Raised when a data acquisition cannot safely continue."""


@dataclass(frozen=True, slots=True)
class WikipediaAcquisitionRequest:
    """Immutable parameters that identify one acquisition smoke run."""

    source: DataSource
    config_name: str
    split: str
    limit: int

    def __post_init__(self) -> None:
        if self.source.acquisition.provider != "huggingface":
            raise AcquisitionError("Wikipedia source must use the huggingface provider")
        if not self.config_name:
            raise AcquisitionError("config_name must not be empty")
        if not self.split:
            raise AcquisitionError("split must not be empty")
        if self.limit <= 0:
            raise AcquisitionError("limit must be positive")

    def identity(self) -> dict[str, Any]:
        return {
            "source_id": self.source.source_id,
            "dataset": self.source.acquisition.location,
            "revision": self.source.acquisition.revision,
            "config_name": self.config_name,
            "split": self.split,
            "limit": self.limit,
        }


def classify_chinese_script(text: str) -> str:
    """Classify Chinese text as simplified, traditional, or unresolved Chinese."""
    simplified_changes = sum(
        original != converted
        for original, converted in zip(
            text, _SIMPLIFIED_TO_TRADITIONAL.convert(text), strict=True
        )
    )
    traditional_changes = sum(
        original != converted
        for original, converted in zip(
            text, _TRADITIONAL_TO_SIMPLIFIED.convert(text), strict=True
        )
    )
    if simplified_changes > traditional_changes:
        return "zh-Hans"
    if traditional_changes > simplified_changes:
        return "zh-Hant"
    return "zh"


def normalize_wikipedia_record(
    record: Mapping[str, Any], source_id: str
) -> CanonicalDocument:
    """Convert one Hugging Face Wikipedia row to the canonical schema."""
    source_record_id = str(record.get("id", "")).strip()
    text = record.get("text")
    title = record.get("title")
    url = record.get("url")
    if not source_record_id:
        raise DataSchemaError("Wikipedia record is missing a stable id")
    if not isinstance(text, str) or not text.strip():
        raise DataSchemaError(f"Wikipedia record {source_record_id!r} has empty text")
    if not isinstance(title, str) or not isinstance(url, str):
        raise DataSchemaError(
            f"Wikipedia record {source_record_id!r} has invalid metadata"
        )

    return CanonicalDocument.from_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "document_id": make_document_id(source_id, source_record_id),
            "source_id": source_id,
            "source_record_id": source_record_id,
            "text": text,
            "language": classify_chinese_script(text),
            "content_type": "encyclopedia",
            "privacy_warnings": [],
            "quality_warnings": [],
            "metadata": {"title": title, "url": url},
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionError(f"cannot read acquisition state: {path.name}") from error
    if not isinstance(value, dict):
        raise AcquisitionError(f"acquisition state is not an object: {path.name}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    temporary_path.write_text(f"{serialized}\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_state(request: WikipediaAcquisitionRequest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **request.identity(),
        "records_written": 0,
        "completed": False,
    }


def _load_or_create_state(
    request: WikipediaAcquisitionRequest,
    documents_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    expected = _initial_state(request)
    if not state_path.exists():
        if documents_path.exists():
            raise AcquisitionError(
                "documents.jsonl exists without state.json; refusing unsafe overwrite"
            )
        _write_json_atomic(state_path, expected)
        return expected

    state = _read_json(state_path)
    for key in ("schema_version", *request.identity()):
        if state.get(key) != expected[key]:
            raise AcquisitionError(
                f"existing state does not match request field: {key}"
            )
    records_written = state.get("records_written")
    if type(records_written) is not int or records_written < 0:
        raise AcquisitionError("state records_written must be a non-negative integer")
    if _count_lines(documents_path) != records_written:
        raise AcquisitionError("documents.jsonl line count does not match state")
    return state


def acquire_wikipedia_records(
    records: Iterable[Mapping[str, Any]],
    request: WikipediaAcquisitionRequest,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Normalize up to ``limit`` rows with durable per-record resume state."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    documents_path = output_path / "documents.jsonl"
    state_path = output_path / "state.json"
    manifest_path = output_path / "manifest.json"

    state = _load_or_create_state(request, documents_path, state_path)
    records_written = state["records_written"]
    if state.get("completed") is True:
        if not manifest_path.is_file():
            raise AcquisitionError("completed state is missing manifest.json")
        return _read_json(manifest_path)

    if records_written > request.limit:
        raise AcquisitionError("state contains more records than the requested limit")

    selected_records = islice(records, records_written, request.limit)
    mode = "a" if records_written else "w"
    with documents_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for record in selected_records:
            document = normalize_wikipedia_record(record, request.source.source_id)
            handle.write(f"{document.to_json_line()}\n")
            handle.flush()
            os.fsync(handle.fileno())
            records_written += 1
            state["records_written"] = records_written
            _write_json_atomic(state_path, state)

    if records_written != request.limit:
        raise AcquisitionError(
            f"source ended after {records_written} records; expected {request.limit}"
        )

    language_counts: Counter[str] = Counter()
    privacy_counts: Counter[str] = Counter()
    with documents_path.open(encoding="utf-8") as handle:
        for line in handle:
            document = CanonicalDocument.from_json_line(line)
            language_counts[document.language] += 1
            privacy_counts.update(document.privacy_warnings)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        **request.identity(),
        "purpose": "acquisition_smoke",
        "source_enabled": request.source.enabled,
        "record_count": records_written,
        "documents_file": documents_path.name,
        "documents_bytes": documents_path.stat().st_size,
        "documents_sha256": _sha256(documents_path),
        "language_counts": dict(sorted(language_counts.items())),
        "privacy_warning_counts": dict(sorted(privacy_counts.items())),
        "privacy_action": "warn",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(manifest_path, manifest)
    state["completed"] = True
    _write_json_atomic(state_path, state)
    return manifest


def stream_huggingface_wikipedia(
    request: WikipediaAcquisitionRequest, token: str | None
) -> Iterable[Mapping[str, Any]]:
    """Create a streaming iterable for the pinned Hugging Face dataset."""
    from datasets import load_dataset

    dataset = load_dataset(
        request.source.acquisition.location,
        request.config_name,
        split=request.split,
        revision=request.source.acquisition.revision,
        streaming=True,
        token=token,
    )
    return dataset


def _find_source(registry_path: Path, source_id: str) -> DataSource:
    registry = load_source_registry(registry_path)
    for source in registry.sources:
        if source.source_id == source_id:
            return source
    raise AcquisitionError(f"source is not registered: {source_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire the pinned Wikipedia Chinese smoke sample."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/data/sources.yaml"),
    )
    parser.add_argument("--source-id", default="wikipedia-20231101")
    parser.add_argument("--config-name", default="20231101.zh")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-smoke"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    args = build_parser().parse_args(argv)
    source = _find_source(args.registry, args.source_id)
    request = WikipediaAcquisitionRequest(
        source=source,
        config_name=args.config_name,
        split=args.split,
        limit=args.limit,
    )
    records = stream_huggingface_wikipedia(request, os.getenv("HF_TOKEN"))
    manifest = acquire_wikipedia_records(records, request, args.output_dir)
    print(
        "Wikipedia acquisition complete: "
        f"{manifest['record_count']} records, "
        f"sha256={manifest['documents_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
