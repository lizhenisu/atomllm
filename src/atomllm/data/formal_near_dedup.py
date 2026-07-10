"""Disk-backed conservative SimHash near-duplicate clustering for formal data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import random
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.formal_exact_dedup import _read_json, _sha256, _write_json_atomic
from atomllm.data.formal_acquisition import estimate_tokens
from atomllm.data.schema import CanonicalDocument


NEAR_DEDUP_VERSION = "formal-70g-near-dedup-v8"
DEFAULT_EXACT_DIR = Path("artifacts/data/formal-70g/exact-dedup-v1")
DEFAULT_CLEAN_DIR = Path("artifacts/data/formal-70g/clean-v1")
DEFAULT_FINGERPRINT_DIR = Path("artifacts/data/formal-70g/fingerprint-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/data/formal-70g/near-dedup-v8")
# A 16-bit band creates roughly billions of candidate pairs at 14M documents.
# Instead, each probe retains a deterministic random half (32 bits) of the
# SimHash.  A pair at Hamming distance <= 3 collides in one probe with about
# 12% probability; 8 independent probes make this a deliberately
# conservative detector while keeping the external index (about 111M rows)
# feasible on the project's local disk and memory budget.
PROBE_COUNT = 8
PROBE_WIDTH_BITS = 32
MAX_HAMMING_DISTANCE = 3
PROBE_INSERT_BATCH_SIZE = 100_000
DEFAULT_WORKERS = 3


class FormalNearDedupError(RuntimeError):
    """Raised when the disk-backed near-dedup graph cannot be verified."""


def _connect(path: Path, *, temporary: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    if temporary:
        # Per-worker probes are disposable: only a successful merge into the
        # final database becomes part of the formal data version.  Disabling
        # per-transaction journalling removes the dominant I/O bottleneck.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    else:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _commit_temporary(connection: sqlite3.Connection) -> None:
    """Commit a disposable worker database, tolerating transient lock races."""
    for attempt in range(8):
        try:
            connection.commit()
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower() or attempt == 7:
                raise
            time.sleep(0.5 * (attempt + 1))


def _distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _probe_masks() -> tuple[int, ...]:
    """Return versioned, deterministic 32-bit bit selections for LSH."""
    generator = random.Random("atom-formal-70g-near-dedup-v3")
    masks: list[int] = []
    while len(masks) < PROBE_COUNT:
        mask = 0
        for bit in generator.sample(range(64), PROBE_WIDTH_BITS):
            mask |= 1 << bit
        if mask not in masks:
            masks.append(mask)
    return tuple(masks)


def _length_bucket(document: CanonicalDocument) -> str:
    tokens = estimate_tokens(document.text, document.language)
    for upper in (256, 512, 1024, 2048, 4096, 8192):
        if tokens < upper:
            return f"lt-{upper}"
    return "gte-8192"


def _create_probe_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS probes (
          bucket TEXT NOT NULL,
          probe INTEGER NOT NULL,
          signature TEXT NOT NULL,
          document_id TEXT NOT NULL,
          simhash TEXT NOT NULL,
          PRIMARY KEY (bucket, probe, signature, document_id)
        ) WITHOUT ROWID
        """
    )


def _prepare_location_database(location_database: Path, exact_database: Path) -> None:
    """Materialize an indexed representative-location map without mutating exact DB."""
    connection = _connect(location_database)
    try:
        connection.execute("ATTACH DATABASE ? AS exactdb", (str(exact_database),))
        connection.executescript(
            """
            CREATE TABLE representatives (
              shard_name TEXT NOT NULL,
              line_number INTEGER NOT NULL,
              document_id TEXT NOT NULL,
              PRIMARY KEY (shard_name, line_number)
            ) WITHOUT ROWID;
            INSERT INTO representatives
              SELECT shard_name, line_number, document_id
              FROM exactdb.exact_representatives;
            """
        )
        connection.commit()
    finally:
        connection.close()


def _prepare_worker_location_databases(
    worker_dir: Path, exact_database: Path, shard_groups: list[list[str]]
) -> list[Path]:
    """Give each worker a local representative map to avoid shared-read I/O."""
    master = worker_dir / "representative-locations.sqlite3"
    _prepare_location_database(master, exact_database)
    locations: list[Path] = []
    for index, shard_group in enumerate(shard_groups):
        location = worker_dir / f"representative-locations-{index:02d}.sqlite3"
        connection = _connect(location)
        try:
            connection.execute("ATTACH DATABASE ? AS masterdb", (str(master),))
            connection.execute(
                """
                CREATE TABLE representatives (
                  shard_name TEXT NOT NULL,
                  line_number INTEGER NOT NULL,
                  document_id TEXT NOT NULL,
                  PRIMARY KEY (shard_name, line_number)
                ) WITHOUT ROWID
                """
            )
            fingerprint_names = [f"{Path(name).stem}.fingerprints.jsonl" for name in shard_group]
            placeholders = ",".join("?" for _ in fingerprint_names)
            connection.execute(
                "INSERT INTO representatives "
                "SELECT shard_name, line_number, document_id FROM masterdb.representatives "
                f"WHERE shard_name IN ({placeholders})",
                fingerprint_names,
            )
            connection.commit()
        finally:
            connection.close()
        locations.append(location)
    return locations


def _build_probe_worker(spec: tuple[str, str, str, str, list[str], int]) -> str:
    """Build one independent probe database for a disjoint shard group."""
    location_database, clean_directory, fingerprint_directory, worker_database, shards, _ = spec
    output = Path(worker_database)
    connection = _connect(output, temporary=True)
    locations = sqlite3.connect(f"file:{location_database}?mode=ro", uri=True)
    try:
        _create_probe_table(connection)
        masks = _probe_masks()
        batch: list[tuple[str, int, str, str, str]] = []
        for shard_name in shards:
            fingerprint_name = f"{Path(shard_name).stem}.fingerprints.jsonl"
            reps = {
                line_number: document_id
                for line_number, document_id in locations.execute(
                    "SELECT line_number, document_id FROM representatives "
                    "WHERE shard_name=?",
                    (fingerprint_name,),
                )
            }
            with (
                (Path(clean_directory) / "shards" / shard_name).open(encoding="utf-8") as handle,
                (Path(fingerprint_directory) / "shards" / fingerprint_name).open(
                    encoding="utf-8"
                ) as fingerprint_handle,
            ):
                for line_number, (line, fingerprint_line) in enumerate(
                    zip(handle, fingerprint_handle, strict=True)
                ):
                    document_id = reps.get(line_number)
                    if document_id is None:
                        continue
                    document = CanonicalDocument.from_json_line(line)
                    simhash = json.loads(fingerprint_line)["simhash"]
                    bucket = f"{document.language}|{document.content_type}|{_length_bucket(document)}"
                    value = int(simhash, 16)
                    for probe, mask in enumerate(masks):
                        batch.append(
                            (bucket, probe, f"{value & mask:016x}", document_id, simhash)
                        )
                    if len(batch) >= PROBE_INSERT_BATCH_SIZE:
                        connection.executemany("INSERT INTO probes VALUES (?, ?, ?, ?, ?)", batch)
                        _commit_temporary(connection)
                        batch.clear()
        if batch:
            connection.executemany("INSERT INTO probes VALUES (?, ?, ?, ?, ?)", batch)
            _commit_temporary(connection)
        return str(output)
    finally:
        locations.close()
        connection.close()


def _partition_shards(shards: list[str], workers: int) -> list[list[str]]:
    return [shards[index::workers] for index in range(workers) if shards[index::workers]]


def near_deduplicate_formal(
    *,
    project_root: str | Path = ".",
    exact_dir: str | Path = DEFAULT_EXACT_DIR,
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Cluster exact representatives with an external SimHash candidate graph."""
    root = Path(project_root)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise FormalNearDedupError("workers must be an integer from 1 through 4")
    exact_path = root / exact_dir
    exact_manifest_path = exact_path / "manifest.json"
    if not exact_manifest_path.is_file():
        raise FormalNearDedupError("exact dedup manifest.json is missing")
    exact_manifest = _read_json(exact_manifest_path)
    database_name = exact_manifest.get("database_file")
    if not isinstance(database_name, str):
        raise FormalNearDedupError("exact dedup database_file is invalid")
    exact_database = exact_path / database_name
    if _sha256(exact_database) != exact_manifest.get("database_sha256"):
        raise FormalNearDedupError("exact dedup database SHA-256 mismatch")
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    identity = {
        "near_dedup_version": NEAR_DEDUP_VERSION,
        "exact_manifest_sha256": _sha256(exact_manifest_path),
        "clean_manifest_sha256": _sha256(root / clean_dir / "manifest.json"),
    }
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing
        raise FormalNearDedupError(
            "existing near dedup manifest has different input"
        )
    database = output_path / "near-dedup.sqlite3"
    connection = _connect(database)
    try:
        connection.execute("ATTACH DATABASE ? AS exactdb", (str(exact_database),))
        _create_probe_table(connection)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS near_edges (left_id TEXT NOT NULL, right_id TEXT NOT NULL, PRIMARY KEY(left_id, right_id))"
        )
        masks = _probe_masks()
        if connection.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0:
            clean_path = root / clean_dir
            fingerprint_path = root / fingerprint_dir
            clean_manifest = _read_json(clean_path / "manifest.json")
            shard_names = [item["name"] for item in clean_manifest["shards"]]
            worker_dir = output_path / "probe-workers"
            worker_dir.mkdir(exist_ok=True)
            shard_groups = _partition_shards(shard_names, workers)
            location_databases = _prepare_worker_location_databases(
                worker_dir, exact_database, shard_groups
            )
            worker_specs = [
                (
                    str(location_database),
                    str(clean_path),
                    str(fingerprint_path),
                    str(worker_dir / f"worker-{index:02d}.sqlite3"),
                    shard_group,
                    index,
                )
                for index, (shard_group, location_database) in enumerate(
                    zip(shard_groups, location_databases, strict=True)
                )
            ]
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(worker_specs), mp_context=context) as executor:
                worker_databases = list(executor.map(_build_probe_worker, worker_specs))
            for worker_database in worker_databases:
                connection.execute("ATTACH DATABASE ? AS workerdb", (worker_database,))
                connection.execute(
                    "INSERT INTO probes SELECT bucket, probe, signature, document_id, simhash FROM workerdb.probes"
                )
                connection.commit()
                connection.execute("DETACH DATABASE workerdb")
        candidates = connection.execute(
            "SELECT a.document_id, a.simhash, b.document_id, b.simhash "
            "FROM probes a JOIN probes b "
            "ON a.bucket=b.bucket AND a.probe=b.probe AND a.signature=b.signature "
            "AND a.document_id<b.document_id"
        )
        batch = []
        for left_id, left_hash, right_id, right_hash in candidates:
            if _distance(left_hash, right_hash) <= MAX_HAMMING_DISTANCE:
                batch.append((left_id, right_id))
            if len(batch) >= 10000:
                connection.executemany(
                    "INSERT OR IGNORE INTO near_edges VALUES (?, ?)", batch
                )
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO near_edges VALUES (?, ?)", batch
            )
            connection.commit()
        edge_count = connection.execute("SELECT COUNT(*) FROM near_edges").fetchone()[0]
        connection.executescript(
            """
            DROP TABLE IF EXISTS near_components;
            CREATE TABLE near_components AS
            WITH RECURSIVE
              nodes(document_id) AS (SELECT left_id FROM near_edges UNION SELECT right_id FROM near_edges),
              directed(left_id, right_id) AS (SELECT left_id, right_id FROM near_edges UNION SELECT right_id, left_id FROM near_edges),
              reachable(root, document_id) AS (
                SELECT document_id, document_id FROM nodes
                UNION
                SELECT reachable.root, directed.right_id FROM reachable JOIN directed ON directed.left_id = reachable.document_id
              )
            SELECT document_id, MIN(root) AS component_id FROM reachable GROUP BY document_id;
            CREATE INDEX near_components_component_id ON near_components(component_id);
            """
        )
        clusters_path = output_path / "near-duplicate-clusters.jsonl"
        temporary = clusters_path.with_suffix(".jsonl.tmp")
        cluster_count = 0
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            cursor = connection.execute(
                "SELECT component_id, document_id FROM near_components "
                "ORDER BY component_id, document_id"
            )
            component_id: str | None = None
            members: list[str] = []
            for current_component, document_id in cursor:
                if component_id is not None and current_component != component_id:
                    handle.write(
                        json.dumps(
                            {
                                "cluster_id": f"near-{component_id}",
                                "kind": "near",
                                "representative_document_id": members[0],
                                "members": members,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    cluster_count += 1
                    members = []
                component_id = current_component
                members.append(document_id)
            if members and component_id is not None:
                handle.write(
                    json.dumps(
                        {
                            "cluster_id": f"near-{component_id}",
                            "kind": "near",
                            "representative_document_id": members[0],
                            "members": members,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                cluster_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, clusters_path)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {
            **identity,
            "representative_count": exact_manifest.get("representative_count"),
            "lsh": {
                "method": "deterministic-random-32-bit-probes",
                "probe_count": PROBE_COUNT,
                "probe_width_bits": PROBE_WIDTH_BITS,
                "workers": workers,
                "bucket_key": "language|content_type|estimated_token_length_bucket",
                "probe_masks": [f"{mask:016x}" for mask in masks],
                "max_hamming_distance": MAX_HAMMING_DISTANCE,
            },
            "near_duplicate_pair_count": edge_count,
            "near_cluster_count": cluster_count,
            "clusters_file": clusters_path.name,
            "clusters_sha256": _sha256(clusters_path),
            "database_file": database.name,
            "database_sha256": _sha256(database),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="External near deduplication for formal data."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--exact-dir", type=Path, default=DEFAULT_EXACT_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--fingerprint-dir", type=Path, default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    manifest = near_deduplicate_formal(
        project_root=args.project_root,
        exact_dir=args.exact_dir,
        clean_dir=args.clean_dir,
        fingerprint_dir=args.fingerprint_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(
        json.dumps(
            {"near_duplicate_pair_count": manifest["near_duplicate_pair_count"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
