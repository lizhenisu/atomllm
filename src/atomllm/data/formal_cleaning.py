"""Bounded-memory cleaning and sharding for a formal data acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from atomllm.data.cleaning import CLEANING_VERSION, clean_document
from atomllm.data.schema import CanonicalDocument, SCHEMA_VERSION


FORMAL_CLEANING_VERSION = "formal-70g-clean-v1"
DEFAULT_INPUT_DIR = Path("artifacts/data/formal-70g/acquired-space-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/data/formal-70g/clean-v1")
DEFAULT_RECORDS_PER_SHARD = 100_000


class FormalCleaningError(RuntimeError):
    """Raised when a formal cleaning run cannot safely resume."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalCleaningError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalCleaningError(f"JSON must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def _input_identity(input_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = input_dir / "manifest.json"
    documents_path = input_dir / "documents.jsonl"
    if not manifest_path.is_file() or not documents_path.is_file():
        raise FormalCleaningError(
            "input must contain manifest.json and documents.jsonl"
        )
    manifest = _read_json(manifest_path)
    digest, lines = _sha256_and_lines(documents_path)
    if digest != manifest.get("documents_sha256"):
        raise FormalCleaningError("input documents SHA-256 mismatch")
    if lines != manifest.get("record_count"):
        raise FormalCleaningError("input document line count mismatch")
    return documents_path, {
        "input_documents_sha256": digest,
        "input_manifest_sha256": _sha256_and_lines(manifest_path)[0],
        "input_record_count": lines,
    }


def _initial_state(identity: dict[str, Any], records_per_shard: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cleaning_version": FORMAL_CLEANING_VERSION,
        "transform_version": CLEANING_VERSION,
        **identity,
        "records_per_shard": records_per_shard,
        "records_written": 0,
        "completed": False,
    }


def _load_state(
    output_dir: Path, identity: dict[str, Any], records_per_shard: int
) -> dict[str, Any]:
    path = output_dir / "state.json"
    expected = _initial_state(identity, records_per_shard)
    if not path.exists():
        if any((output_dir / "shards").glob("*.jsonl")):
            raise FormalCleaningError("shards exist without state.json")
        _write_json_atomic(path, expected)
        return expected
    state = _read_json(path)
    for key, value in expected.items():
        if key in {"records_written", "completed"}:
            continue
        if state.get(key) != value:
            raise FormalCleaningError(f"state does not match input: {key}")
    if type(state.get("records_written")) is not int or state["records_written"] < 0:
        raise FormalCleaningError("state records_written is invalid")
    return state


def _shard_path(shard_dir: Path, index: int) -> Path:
    return shard_dir / f"part-{index:05d}.jsonl"


def clean_formal(
    *,
    project_root: str | Path = ".",
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    records_per_shard: int = DEFAULT_RECORDS_PER_SHARD,
) -> dict[str, Any]:
    """Clean a formal acquisition into durable fixed-record JSONL shards."""
    if type(records_per_shard) is not int or records_per_shard <= 0:
        raise FormalCleaningError("records_per_shard must be positive")
    root = Path(project_root)
    input_path, identity = _input_identity(root / input_dir)
    output_path = root / output_dir
    shard_dir = output_path / "shards"
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    state = _load_state(output_path, identity, records_per_shard)
    manifest_path = output_path / "manifest.json"
    if state["completed"]:
        if not manifest_path.is_file():
            raise FormalCleaningError("completed state is missing manifest.json")
        return _read_json(manifest_path)

    written = state["records_written"]
    if written > identity["input_record_count"]:
        raise FormalCleaningError("state exceeds input record count")
    if written % records_per_shard:
        raise FormalCleaningError("state must end at a completed shard boundary")
    shard_index = written // records_per_shard
    with input_path.open(encoding="utf-8") as handle:
        iterator = islice(handle, written, None)
        while written < identity["input_record_count"]:
            target = min(records_per_shard, identity["input_record_count"] - written)
            final_path = _shard_path(shard_dir, shard_index)
            if final_path.exists():
                raise FormalCleaningError(
                    f"refusing to overwrite shard: {final_path.name}"
                )
            temporary = final_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                for line in islice(iterator, target):
                    document = CanonicalDocument.from_json_line(line)
                    output.write(
                        f"{clean_document(document).document.to_json_line()}\n"
                    )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, final_path)
            actual_lines = _sha256_and_lines(final_path)[1]
            if actual_lines != target:
                raise FormalCleaningError("written shard line count mismatch")
            written += target
            shard_index += 1
            state["records_written"] = written
            _write_json_atomic(output_path / "state.json", state)

    shards = sorted(shard_dir.glob("part-*.jsonl"))
    records = 0
    quality: Counter[str] = Counter()
    privacy: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    shard_metadata: list[dict[str, Any]] = []
    for shard in shards:
        digest, lines = _sha256_and_lines(shard)
        shard_metadata.append(
            {"name": shard.name, "sha256": digest, "record_count": lines}
        )
        records += lines
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                document = CanonicalDocument.from_json_line(line)
                languages[document.language] += 1
                quality.update(document.quality_warnings)
                privacy.update(document.privacy_warnings)
    if records != identity["input_record_count"]:
        raise FormalCleaningError("cleaned shards do not cover all input records")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cleaning_version": FORMAL_CLEANING_VERSION,
        "transform_version": CLEANING_VERSION,
        **identity,
        "record_count": records,
        "retained_count": records,
        "dropped_count": 0,
        "shard_count": len(shard_metadata),
        "shards": shard_metadata,
        "language_counts": dict(sorted(languages.items())),
        "quality_warning_counts": dict(sorted(quality.items())),
        "privacy_warning_counts": dict(sorted(privacy.items())),
        "privacy_action": "warn",
        "quality_action": "warn",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(manifest_path, manifest)
    state["completed"] = True
    _write_json_atomic(output_path / "state.json", state)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream-clean and shard formal data."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--records-per-shard", type=int, default=DEFAULT_RECORDS_PER_SHARD
    )
    args = parser.parse_args(argv)
    manifest = clean_formal(
        project_root=args.project_root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        records_per_shard=args.records_per_shard,
    )
    print(
        json.dumps(
            {
                "record_count": manifest["record_count"],
                "shard_count": manifest["shard_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
