"""Streaming fingerprint shards for external formal-data deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.deduplication import _character_shingles, _simhash
from atomllm.data.schema import CanonicalDocument, SCHEMA_VERSION


FINGERPRINT_VERSION = "formal-70g-fingerprint-v1"
DEFAULT_CLEAN_DIR = Path("artifacts/data/formal-70g/clean-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/data/formal-70g/fingerprint-v1")


class FormalFingerprintError(RuntimeError):
    """Raised when fingerprint inputs or resumable output are inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalFingerprintError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalFingerprintError(f"JSON must be an object: {path}")
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


def _fingerprint(document: CanonicalDocument) -> dict[str, Any]:
    text_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    shingles = _character_shingles(document.text)
    return {
        "document_id": document.document_id,
        "text_sha256": text_hash,
        "character_count": len(document.text),
        "simhash": f"{_simhash(shingles):016x}",
    }


def fingerprint_formal(
    *,
    project_root: str | Path = ".",
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Create one durable fingerprint shard per verified cleaned text shard."""
    root = Path(project_root)
    clean_path = root / clean_dir
    clean_manifest_path = clean_path / "manifest.json"
    if not clean_manifest_path.is_file():
        raise FormalFingerprintError("clean manifest.json is missing")
    clean_manifest = _read_json(clean_manifest_path)
    raw_shards = clean_manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise FormalFingerprintError("clean manifest has invalid shards")

    output_path = root / output_dir
    shard_dir = output_path / "shards"
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    manifest_path = output_path / "manifest.json"
    state_path = output_path / "state.json"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "clean_manifest_sha256": _sha256_and_lines(clean_manifest_path)[0],
        "clean_documents_sha256": clean_manifest.get("input_documents_sha256"),
    }
    if not isinstance(identity["clean_documents_sha256"], str):
        raise FormalFingerprintError("clean manifest has invalid input identity")
    if state_path.exists():
        state = _read_json(state_path)
        for key, value in identity.items():
            if state.get(key) != value:
                raise FormalFingerprintError(f"state identity mismatch: {key}")
    else:
        state = {**identity, "completed_shards": 0, "completed": False}
        _write_json_atomic(state_path, state)
    if state.get("completed") is True:
        if not manifest_path.is_file():
            raise FormalFingerprintError("completed state is missing manifest.json")
        return _read_json(manifest_path)
    completed = state.get("completed_shards")
    if type(completed) is not int or completed < 0 or completed > len(raw_shards):
        raise FormalFingerprintError("state completed_shards is invalid")

    output_shards: list[dict[str, Any]] = []
    for index, metadata in enumerate(raw_shards):
        if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
            raise FormalFingerprintError("clean shard metadata is invalid")
        source_path = clean_path / "shards" / metadata["name"]
        source_digest, source_lines = _sha256_and_lines(source_path)
        if source_digest != metadata.get("sha256") or source_lines != metadata.get(
            "record_count"
        ):
            raise FormalFingerprintError(
                f"clean shard integrity mismatch: {source_path.name}"
            )
        target_path = shard_dir / f"{Path(metadata['name']).stem}.fingerprints.jsonl"
        if index < completed:
            digest, lines = _sha256_and_lines(target_path)
            output_shards.append(
                {"name": target_path.name, "sha256": digest, "record_count": lines}
            )
            if lines != source_lines:
                raise FormalFingerprintError(
                    "completed fingerprint line count mismatch"
                )
            continue
        temporary = target_path.with_suffix(".jsonl.tmp")
        with (
            source_path.open(encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8", newline="\n") as target,
        ):
            for line in source:
                target.write(
                    json.dumps(
                        _fingerprint(CanonicalDocument.from_json_line(line)),
                        sort_keys=True,
                    )
                    + "\n"
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, target_path)
        digest, lines = _sha256_and_lines(target_path)
        if lines != source_lines:
            raise FormalFingerprintError("fingerprint line count mismatch")
        output_shards.append(
            {"name": target_path.name, "sha256": digest, "record_count": lines}
        )
        state["completed_shards"] = index + 1
        _write_json_atomic(state_path, state)

    manifest = {
        **identity,
        "record_count": sum(item["record_count"] for item in output_shards),
        "shards": output_shards,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(manifest_path, manifest)
    state["completed"] = True
    _write_json_atomic(state_path, state)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create formal-data fingerprint shards."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = fingerprint_formal(
        project_root=args.project_root,
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "record_count": manifest["record_count"],
                "shard_count": len(manifest["shards"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
