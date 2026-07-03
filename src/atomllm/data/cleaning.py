"""Deterministic text cleaning and quality warning generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from atomllm.data.schema import SCHEMA_VERSION, CanonicalDocument


CLEANING_VERSION = "clean-v1"
MIN_DOCUMENT_CHARS = 200
MAX_DOCUMENT_CHARS = 50_000
MAX_CONSECUTIVE_BLANK_LINES = 1
HIGH_REPETITION_RATIO = 0.2
MIN_REPETITION_PARAGRAPH_CHARS = 20
REPEATED_CHARACTER_RUN = 20

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_CHARACTER = re.compile(r"(.)\1{19,}", re.DOTALL)


class CleaningError(RuntimeError):
    """Raised when a cleaning run cannot safely continue."""


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """One cleaned document and the deterministic operations applied to it."""

    document: CanonicalDocument
    changes: tuple[str, ...]


def normalize_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Apply the versioned, semantics-preserving clean-v1 transformations."""
    changes: list[str] = []
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        changes.append("unicode_nfc")

    line_normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if line_normalized != normalized:
        changes.append("line_endings")

    trimmed_lines = "\n".join(line.rstrip() for line in line_normalized.split("\n"))
    if trimmed_lines != line_normalized:
        changes.append("trailing_whitespace")

    collapsed = _EXCESS_BLANK_LINES.sub("\n\n", trimmed_lines)
    if collapsed != trimmed_lines:
        changes.append("excess_blank_lines")

    stripped = collapsed.strip()
    if stripped != collapsed:
        changes.append("outer_whitespace")
    return stripped, tuple(changes)


def _repeated_paragraph_ratio(text: str) -> float:
    paragraphs = [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) >= MIN_REPETITION_PARAGRAPH_CHARS
    ]
    if len(paragraphs) < 5:
        return 0.0
    seen: set[str] = set()
    repeated_chars = 0
    total_chars = 0
    for paragraph in paragraphs:
        total_chars += len(paragraph)
        if paragraph in seen:
            repeated_chars += len(paragraph)
        else:
            seen.add(paragraph)
    return repeated_chars / total_chars if total_chars else 0.0


def quality_warnings(text: str, language: str) -> tuple[str, ...]:
    """Generate conservative warnings without rejecting the document."""
    warnings: list[str] = []
    if len(text) < MIN_DOCUMENT_CHARS:
        warnings.append("too_short")
    if len(text) > MAX_DOCUMENT_CHARS:
        warnings.append("too_long")
    if "\ufffd" in text:
        warnings.append("decode_replacement")
    if (
        _REPEATED_CHARACTER.search(text)
        or _repeated_paragraph_ratio(text) >= HIGH_REPETITION_RATIO
    ):
        warnings.append("high_repetition")
    if language == "zh":
        warnings.append("low_language_confidence")
    return tuple(warnings)


def clean_document(document: CanonicalDocument) -> CleaningResult:
    """Clean one canonical document while preserving its stable identity."""
    text, changes = normalize_text(document.text)
    generated_warnings = quality_warnings(text, document.language)
    merged_warnings = tuple(
        dict.fromkeys((*document.quality_warnings, *generated_warnings))
    )
    cleaned = CanonicalDocument.from_mapping(
        {
            **document.to_mapping(),
            "text": text,
            "quality_warnings": list(merged_warnings),
        }
    )
    return CleaningResult(document=cleaned, changes=changes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CleaningError(f"cannot read JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise CleaningError(f"JSON file is not an object: {path.name}")
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


def _input_identity(input_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    input_manifest_path = input_dir / "manifest.json"
    input_documents_path = input_dir / "documents.jsonl"
    if not input_manifest_path.is_file() or not input_documents_path.is_file():
        raise CleaningError(
            "input directory must contain manifest.json and documents.jsonl"
        )

    input_manifest = _read_json(input_manifest_path)
    record_count = input_manifest.get("record_count")
    expected_sha256 = input_manifest.get("documents_sha256")
    if type(record_count) is not int or record_count <= 0:
        raise CleaningError("input manifest has an invalid record_count")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise CleaningError("input manifest has an invalid documents_sha256")
    actual_sha256 = _sha256(input_documents_path)
    if actual_sha256 != expected_sha256:
        raise CleaningError("input documents SHA-256 does not match its manifest")
    if _count_lines(input_documents_path) != record_count:
        raise CleaningError("input documents line count does not match its manifest")

    identity = {
        "cleaning_version": CLEANING_VERSION,
        "input_record_count": record_count,
        "input_documents_sha256": actual_sha256,
        "input_manifest_sha256": _sha256(input_manifest_path),
    }
    return input_documents_path, input_manifest, identity


def _load_or_create_state(
    output_dir: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    documents_path = output_dir / "documents.jsonl"
    expected = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "records_written": 0,
        "completed": False,
    }
    if not state_path.exists():
        if documents_path.exists():
            raise CleaningError(
                "output documents exist without state.json; refusing unsafe overwrite"
            )
        _write_json_atomic(state_path, expected)
        return expected

    state = _read_json(state_path)
    for key in ("schema_version", *identity):
        if state.get(key) != expected[key]:
            raise CleaningError(f"existing state does not match input field: {key}")
    records_written = state.get("records_written")
    if type(records_written) is not int or records_written < 0:
        raise CleaningError("state records_written must be a non-negative integer")
    if _count_lines(documents_path) != records_written:
        raise CleaningError("output line count does not match cleaning state")
    return state


def clean_dataset(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Clean a canonical JSONL dataset with durable per-document resume state."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_documents_path = output_path / "documents.jsonl"
    output_manifest_path = output_path / "manifest.json"
    state_path = output_path / "state.json"

    input_documents_path, input_manifest, identity = _input_identity(input_path)
    state = _load_or_create_state(output_path, identity)
    records_written = state["records_written"]
    if state.get("completed") is True:
        if not output_manifest_path.is_file():
            raise CleaningError("completed state is missing manifest.json")
        return _read_json(output_manifest_path)

    input_count = identity["input_record_count"]
    if records_written > input_count:
        raise CleaningError("state contains more records than the input dataset")

    mode = "a" if records_written else "w"
    with (
        input_documents_path.open(encoding="utf-8") as input_handle,
        output_documents_path.open(
            mode, encoding="utf-8", newline="\n"
        ) as output_handle,
    ):
        for line in islice(input_handle, records_written, None):
            document = CanonicalDocument.from_json_line(line)
            result = clean_document(document)
            output_handle.write(f"{result.document.to_json_line()}\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())
            records_written += 1
            state["records_written"] = records_written
            _write_json_atomic(state_path, state)

    if records_written != input_count:
        raise CleaningError(
            f"cleaning ended after {records_written} records; expected {input_count}"
        )

    language_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    privacy_counts: Counter[str] = Counter()
    changed_documents = 0
    change_counts: Counter[str] = Counter()
    with (
        input_documents_path.open(encoding="utf-8") as input_handle,
        output_documents_path.open(encoding="utf-8") as output_handle,
    ):
        for input_line, output_line in zip(input_handle, output_handle, strict=True):
            before = CanonicalDocument.from_json_line(input_line)
            after = CanonicalDocument.from_json_line(output_line)
            result = clean_document(before)
            if result.document != after:
                raise CleaningError("output document does not match clean-v1 transform")
            if result.changes:
                changed_documents += 1
                change_counts.update(result.changes)
            language_counts[after.language] += 1
            quality_counts.update(after.quality_warnings)
            privacy_counts.update(after.privacy_warnings)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "transform": {
            "name": "deterministic_text_cleaning",
            "version": CLEANING_VERSION,
            "min_document_chars": MIN_DOCUMENT_CHARS,
            "max_document_chars": MAX_DOCUMENT_CHARS,
            "high_repetition_ratio": HIGH_REPETITION_RATIO,
            "min_repetition_paragraph_chars": MIN_REPETITION_PARAGRAPH_CHARS,
            "repeated_character_run": REPEATED_CHARACTER_RUN,
            "privacy_action": "warn",
            "quality_action": "warn",
        },
        "source_id": input_manifest.get("source_id"),
        "record_count": records_written,
        "retained_count": records_written,
        "dropped_count": 0,
        "changed_document_count": changed_documents,
        "change_counts": dict(sorted(change_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "quality_warning_counts": dict(sorted(quality_counts.items())),
        "privacy_warning_counts": dict(sorted(privacy_counts.items())),
        "documents_file": output_documents_path.name,
        "documents_bytes": output_documents_path.stat().st_size,
        "documents_sha256": _sha256(output_documents_path),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(output_manifest_path, manifest)
    state["completed"] = True
    _write_json_atomic(state_path, state)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean the Wikipedia Chinese smoke sample deterministically."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-smoke"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-clean-v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = clean_dataset(args.input_dir, args.output_dir)
    print(
        "Wikipedia cleaning complete: "
        f"{manifest['record_count']} records, "
        f"{manifest['changed_document_count']} changed, "
        f"sha256={manifest['documents_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
