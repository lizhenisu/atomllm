"""Deterministic two-way split of externally deduplicated formal data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.formal_exact_dedup import _read_json, _sha256, _write_json_atomic
from atomllm.data.formal_split_config import load_formal_split_config
from atomllm.data.formal_acquisition import estimate_tokens
from atomllm.data.schema import CanonicalDocument, SCHEMA_VERSION


SPLIT_VERSION = "formal-70g-split-v1"
DEFAULT_CLEAN_DIR = Path("artifacts/data/formal-70g/clean-v1")
DEFAULT_EXACT_DIR = Path("artifacts/data/formal-70g/exact-dedup-v1")
DEFAULT_NEAR_DIR = Path("artifacts/data/formal-70g/near-dedup-v8")
DEFAULT_OUTPUT_DIR = Path("artifacts/data/formal-70g/split-v1")
DEFAULT_CONFIG = Path("configs/data/formal-70g-processing.yaml")


class FormalSplitError(RuntimeError):
    """Raised when the fixed train/validation partition is unsafe."""


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def _manifest_identity(clean: Path, exact: Path, near: Path, config: Path) -> dict[str, str]:
    return {
        "clean_manifest_sha256": _sha256(clean / "manifest.json"),
        "exact_manifest_sha256": _sha256(exact / "manifest.json"),
        "near_manifest_sha256": _sha256(near / "manifest.json"),
        "processing_config_sha256": _sha256(config),
    }


def _prepare_index(
    database: Path, exact_database: Path, near_database: Path
) -> tuple[sqlite3.Connection, int]:
    connection = _connect(database)
    connection.execute("ATTACH DATABASE ? AS exactdb", (str(exact_database),))
    connection.execute("ATTACH DATABASE ? AS neardb", (str(near_database),))
    connection.executescript(
        """
        DROP TABLE IF EXISTS retained;
        CREATE TABLE retained AS
          SELECT r.document_id, r.shard_name, r.line_number
          FROM exactdb.exact_representatives AS r
          LEFT JOIN neardb.near_components AS n USING(document_id)
          WHERE n.document_id IS NULL OR n.component_id = r.document_id;
        CREATE UNIQUE INDEX retained_location ON retained(shard_name, line_number);
        DROP TABLE IF EXISTS candidates;
        CREATE TABLE candidates (
          document_id TEXT PRIMARY KEY,
          stable_rank TEXT NOT NULL
        );
        DROP TABLE IF EXISTS validation_ids;
        CREATE TABLE validation_ids (document_id TEXT PRIMARY KEY);
        """
    )
    count = connection.execute("SELECT COUNT(*) FROM retained").fetchone()[0]
    return connection, count


def _write_shard(path: Path, documents: list[CanonicalDocument]) -> dict[str, Any]:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(f"{document.to_json_line()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest, count = _sha256_and_lines(path)
    return {"name": path.name, "sha256": digest, "record_count": count}


def split_formal(
    *,
    project_root: str | Path = ".",
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
    exact_dir: str | Path = DEFAULT_EXACT_DIR,
    near_dir: str | Path = DEFAULT_NEAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Create the immutable 99/1 split after exact and near deduplication."""
    root = Path(project_root)
    clean_path, exact_path, near_path = root / clean_dir, root / exact_dir, root / near_dir
    config_file = root / config_path
    contract = load_formal_split_config(config_file)
    for directory in (clean_path, exact_path, near_path):
        if not (directory / "manifest.json").is_file():
            raise FormalSplitError(f"required manifest is missing: {directory}")
    clean_manifest = _read_json(clean_path / "manifest.json")
    exact_manifest = _read_json(exact_path / "manifest.json")
    near_manifest = _read_json(near_path / "manifest.json")
    exact_database = exact_path / str(exact_manifest.get("database_file", ""))
    near_database = near_path / str(near_manifest.get("database_file", ""))
    if not exact_database.is_file() or not near_database.is_file():
        raise FormalSplitError("deduplication database is missing")
    if _sha256(exact_database) != exact_manifest.get("database_sha256"):
        raise FormalSplitError("exact dedup database SHA-256 mismatch")
    if _sha256(near_database) != near_manifest.get("database_sha256"):
        raise FormalSplitError("near dedup database SHA-256 mismatch")
    raw_shards = clean_manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise FormalSplitError("clean manifest has invalid shards")

    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    identity = {
        "split_version": SPLIT_VERSION,
        **_manifest_identity(clean_path, exact_path, near_path, config_file),
    }
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in identity.items()):
            return manifest
        raise FormalSplitError("existing split manifest has different input")

    database = output_path / "split-selection.sqlite3"
    connection, retained_count = _prepare_index(database, exact_database, near_database)
    try:
        for shard in raw_shards:
            name = shard.get("name") if isinstance(shard, dict) else None
            if not isinstance(name, str):
                raise FormalSplitError("clean shard metadata is invalid")
            path = clean_path / "shards" / name
            if _sha256(path) != shard.get("sha256"):
                raise FormalSplitError(f"clean shard SHA-256 mismatch: {name}")
            batch: list[tuple[str, str]] = []
            fingerprint_name = f"{Path(name).stem}.fingerprints.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    document = CanonicalDocument.from_json_line(line)
                    if connection.execute(
                        "SELECT 1 FROM retained WHERE shard_name=? AND line_number=?",
                        (fingerprint_name, line_number),
                    ).fetchone() is None:
                        continue
                    if (
                        not document.quality_warnings
                        and contract.min_estimated_tokens
                        <= estimate_tokens(document.text, document.language)
                        <= contract.max_estimated_tokens
                    ):
                        batch.append(
                            (document.document_id, hashlib.sha256(document.document_id.encode()).hexdigest())
                        )
                    if len(batch) >= 10_000:
                        connection.executemany("INSERT INTO candidates VALUES (?, ?)", batch)
                        connection.commit()
                        batch.clear()
            if batch:
                connection.executemany("INSERT INTO candidates VALUES (?, ?)", batch)
                connection.commit()
        candidate_count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        validation_count = round(retained_count * contract.validation_fraction)
        if candidate_count < validation_count:
            raise FormalSplitError(
                f"only {candidate_count} high-quality validation candidates for {validation_count} required documents"
            )
        connection.execute(
            "INSERT INTO validation_ids SELECT document_id FROM candidates ORDER BY stable_rank LIMIT ?",
            (validation_count,),
        )
        connection.commit()

        output_shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        counts: Counter[str] = Counter()
        languages: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
        content_types: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
        for index, shard in enumerate(raw_shards):
            name = shard["name"]
            path = clean_path / "shards" / name
            groups: dict[str, list[CanonicalDocument]] = {"train": [], "validation": []}
            fingerprint_name = f"{Path(name).stem}.fingerprints.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    document = CanonicalDocument.from_json_line(line)
                    if connection.execute(
                        "SELECT 1 FROM retained WHERE shard_name=? AND line_number=?",
                        (fingerprint_name, line_number),
                    ).fetchone() is None:
                        continue
                    split = "validation" if connection.execute(
                        "SELECT 1 FROM validation_ids WHERE document_id=?", (document.document_id,)
                    ).fetchone() else "train"
                    groups[split].append(document)
                    counts[split] += 1
                    languages[split][document.language] += 1
                    content_types[split][document.content_type] += 1
            for split, documents in groups.items():
                if not documents:
                    continue
                directory = output_path / split / "shards"
                directory.mkdir(parents=True, exist_ok=True)
                output_shards[split].append(
                    _write_shard(directory / f"part-{index:05d}.jsonl", documents)
                )
        if counts["train"] + counts["validation"] != retained_count:
            raise FormalSplitError("split output does not cover retained documents")
        if counts["validation"] != validation_count:
            raise FormalSplitError("validation output count mismatch")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {
            **identity,
            "schema_version": SCHEMA_VERSION,
            "training_eligible": True,
            "splits": {"train": counts["train"], "validation": counts["validation"]},
            "retained_after_dedup_count": retained_count,
            "validation_candidate_count": candidate_count,
            "validation_selection": {
                "quality_warnings": contract.validation_quality_warnings,
                "estimated_tokens": [contract.min_estimated_tokens, contract.max_estimated_tokens],
                "stable_tie_breaker": "sha256(document_id)",
            },
            "shards": output_shards,
            "language_counts": {key: dict(sorted(value.items())) for key, value in languages.items()},
            "content_type_counts": {key: dict(sorted(value.items())) for key, value in content_types.items()},
            "selection_database": database.name,
            "selection_database_sha256": _sha256(database),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split formal data into train/validation.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--exact-dir", type=Path, default=DEFAULT_EXACT_DIR)
    parser.add_argument("--near-dir", type=Path, default=DEFAULT_NEAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    manifest = split_formal(
        project_root=args.project_root,
        clean_dir=args.clean_dir,
        exact_dir=args.exact_dir,
        near_dir=args.near_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(json.dumps({"splits": manifest["splits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
