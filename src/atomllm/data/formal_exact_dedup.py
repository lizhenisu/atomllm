"""Disk-backed exact deduplication for formal-data fingerprint shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXACT_DEDUP_VERSION = "formal-70g-exact-dedup-v1"
DEFAULT_FINGERPRINT_DIR = Path("artifacts/data/formal-70g/fingerprint-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/data/formal-70g/exact-dedup-v1")
BATCH_SIZE = 10_000


class FormalExactDedupError(RuntimeError):
    """Raised when the on-disk exact-dedup index is inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalExactDedupError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalExactDedupError(f"JSON must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            document_id TEXT PRIMARY KEY,
            text_sha256 TEXT NOT NULL,
            shard_name TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            simhash TEXT NOT NULL,
            character_count INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS fingerprints_text_sha256 ON fingerprints(text_sha256);
        """
    )
    return connection


def exact_deduplicate_formal(
    *,
    project_root: str | Path = ".",
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build a persistent external exact-dedup index and representative table."""
    root = Path(project_root)
    fingerprints_path = root / fingerprint_dir
    fingerprint_manifest_path = fingerprints_path / "manifest.json"
    if not fingerprint_manifest_path.is_file():
        raise FormalExactDedupError("fingerprint manifest.json is missing")
    fingerprint_manifest = _read_json(fingerprint_manifest_path)
    raw_shards = fingerprint_manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise FormalExactDedupError("fingerprint manifest has invalid shards")
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    identity = {
        "exact_dedup_version": EXACT_DEDUP_VERSION,
        "fingerprint_manifest_sha256": _sha256(fingerprint_manifest_path),
        "fingerprint_record_count": fingerprint_manifest.get("record_count"),
    }
    if type(identity["fingerprint_record_count"]) is not int:
        raise FormalExactDedupError("fingerprint record_count is invalid")
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing
        raise FormalExactDedupError(
            "existing exact dedup manifest has different input"
        )

    database_path = output_path / "exact-dedup.sqlite3"
    connection = _connect(database_path)
    try:
        for metadata in raw_shards:
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("name"), str
            ):
                raise FormalExactDedupError("fingerprint shard metadata is invalid")
            path = fingerprints_path / "shards" / metadata["name"]
            if _sha256(path) != metadata.get("sha256"):
                raise FormalExactDedupError(
                    f"fingerprint shard SHA-256 mismatch: {path.name}"
                )
            batch: list[tuple[str, str, str, int, str, int]] = []
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    item = json.loads(line)
                    batch.append(
                        (
                            item["document_id"],
                            item["text_sha256"],
                            path.name,
                            line_number,
                            item["simhash"],
                            item["character_count"],
                        )
                    )
                    if len(batch) >= BATCH_SIZE:
                        connection.executemany(
                            "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        connection.commit()
                        batch.clear()
            if batch:
                connection.executemany(
                    "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?, ?, ?, ?)",
                    batch,
                )
                connection.commit()
        indexed = connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        if indexed > identity["fingerprint_record_count"]:
            raise FormalExactDedupError("fingerprint database exceeds input count")
        # The source contract makes document_id deterministic from
        # (source_id, source_record_id).  A few upstream streams reuse that
        # identity.  SQLite deliberately keeps the first, shard-ordered
        # occurrence, so the collision becomes an explicit identity-dedup
        # outcome instead of silently being mistaken for a failed import.
        identity_duplicate_count = identity["fingerprint_record_count"] - indexed
        connection.executescript(
            """
            DROP TABLE IF EXISTS exact_representatives;
            CREATE TABLE exact_representatives AS
              SELECT f.document_id, f.shard_name, f.line_number
              FROM fingerprints AS f
              JOIN (
                SELECT MIN(document_id) AS document_id
                FROM fingerprints
                GROUP BY text_sha256
              ) AS selected USING(document_id);
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX exact_representatives_document_id ON exact_representatives(document_id)"
        )
        representative_count = connection.execute(
            "SELECT COUNT(*) FROM exact_representatives"
        ).fetchone()[0]
        exact_duplicate_count = indexed - representative_count
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {
            **identity,
            "record_count": indexed,
            "input_record_count": identity["fingerprint_record_count"],
            "identity_duplicate_document_count": identity_duplicate_count,
            "representative_count": representative_count,
            "exact_duplicate_document_count": exact_duplicate_count,
            "database_file": database_path.name,
            "database_sha256": _sha256(database_path),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="External exact deduplication for formal data."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--fingerprint-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = exact_deduplicate_formal(
        project_root=args.project_root,
        fingerprint_dir=args.fingerprint_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "record_count": manifest["record_count"],
                "exact_duplicate_document_count": manifest[
                    "exact_duplicate_document_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
