"""Deterministic train, validation, and test splitting with leakage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.schema import SCHEMA_VERSION, CanonicalDocument


SPLIT_VERSION = "split-v1"
SPLIT_SALT = "atomllm-wikipedia-split-v1"
SPLIT_RATIOS = {
    "train": 0.98,
    "validation": 0.01,
    "test": 0.01,
}
SPLIT_ORDER = ("train", "validation", "test")


class SplittingError(RuntimeError):
    """Raised when a deterministic split cannot safely continue."""


@dataclass(frozen=True, slots=True)
class InputRecord:
    document: CanonicalDocument
    json_line: str


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


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
        raise SplittingError(f"cannot read JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise SplittingError(f"JSON file is not an object: {path.name}")
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


def _write_lines_atomic(path: Path, lines: Iterable[str]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(f"{line.rstrip(chr(10))}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _load_input(input_dir: Path) -> tuple[list[InputRecord], dict[str, Any]]:
    manifest_path = input_dir / "manifest.json"
    documents_path = input_dir / "documents.jsonl"
    if not manifest_path.is_file() or not documents_path.is_file():
        raise SplittingError(
            "input directory must contain manifest.json and documents.jsonl"
        )
    manifest = _read_json(manifest_path)
    record_count = manifest.get("record_count")
    documents_sha256 = manifest.get("documents_sha256")
    if type(record_count) is not int or record_count <= 0:
        raise SplittingError("input manifest has an invalid record_count")
    if not isinstance(documents_sha256, str) or len(documents_sha256) != 64:
        raise SplittingError("input manifest has an invalid documents_sha256")
    if _sha256(documents_path) != documents_sha256:
        raise SplittingError("input documents SHA-256 does not match its manifest")

    records: list[InputRecord] = []
    seen_ids: set[str] = set()
    with documents_path.open(encoding="utf-8") as handle:
        for line in handle:
            document = CanonicalDocument.from_json_line(line)
            if document.document_id in seen_ids:
                raise SplittingError("input contains a repeated document_id")
            seen_ids.add(document.document_id)
            records.append(InputRecord(document=document, json_line=line.rstrip("\n")))
    if len(records) != record_count:
        raise SplittingError("input documents line count does not match its manifest")
    identity = {
        "split_version": SPLIT_VERSION,
        "input_record_count": record_count,
        "input_documents_sha256": documents_sha256,
        "input_manifest_sha256": _sha256(manifest_path),
    }
    return records, identity


def _load_duplicate_clusters(
    deduplication_dir: Path,
    input_identity: Mapping[str, Any],
    document_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = deduplication_dir / "manifest.json"
    clusters_path = deduplication_dir / "duplicate-clusters.jsonl"
    if not manifest_path.is_file() or not clusters_path.is_file():
        raise SplittingError(
            "deduplication directory must contain manifest.json "
            "and duplicate-clusters.jsonl"
        )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("input_documents_sha256")
        != input_identity["input_documents_sha256"]
    ):
        raise SplittingError(
            "deduplication result belongs to different input documents"
        )
    if manifest.get("record_count") != input_identity["input_record_count"]:
        raise SplittingError("deduplication record count does not match input")
    if _sha256(clusters_path) != manifest.get("clusters_sha256"):
        raise SplittingError("duplicate clusters SHA-256 does not match its manifest")

    clusters: list[dict[str, Any]] = []
    with clusters_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                cluster = json.loads(line)
            except json.JSONDecodeError as error:
                raise SplittingError(
                    "duplicate cluster contains invalid JSON"
                ) from error
            if not isinstance(cluster, dict):
                raise SplittingError("duplicate cluster must be a JSON object")
            members = cluster.get("members")
            if (
                not isinstance(members, list)
                or len(members) < 2
                or not all(isinstance(member, str) for member in members)
            ):
                raise SplittingError("duplicate cluster has invalid members")
            if len(members) != len(set(members)):
                raise SplittingError("duplicate cluster contains repeated members")
            unknown = set(members) - document_ids
            if unknown:
                raise SplittingError("duplicate cluster references an unknown document")
            clusters.append(cluster)
    identity = {
        "deduplication_manifest_sha256": _sha256(manifest_path),
        "duplicate_clusters_sha256": _sha256(clusters_path),
    }
    return clusters, identity


def _target_counts(record_count: int) -> dict[str, int]:
    raw = {split: record_count * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remainder = record_count - sum(counts.values())
    ranked = sorted(
        SPLIT_ORDER,
        key=lambda split: (-(raw[split] - counts[split]), SPLIT_ORDER.index(split)),
    )
    for split in ranked[:remainder]:
        counts[split] += 1
    return counts


def _group_documents(
    document_ids: list[str], clusters: list[dict[str, Any]]
) -> list[tuple[str, ...]]:
    disjoint_set = _DisjointSet(document_ids)
    for cluster in clusters:
        members = cluster["members"]
        for member in members[1:]:
            disjoint_set.union(members[0], member)
    groups: dict[str, list[str]] = defaultdict(list)
    for document_id in document_ids:
        groups[disjoint_set.find(document_id)].append(document_id)
    return [tuple(sorted(members)) for members in groups.values()]


def _rank_group(group: tuple[str, ...], split: str) -> str:
    identity = f"{SPLIT_SALT}\0{split}\0{group[0]}".encode()
    return hashlib.sha256(identity).hexdigest()


def assign_splits(
    document_ids: list[str],
    clusters: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign duplicate-connected groups without crossing split boundaries."""
    targets = _target_counts(len(document_ids))
    groups = _group_documents(document_ids, clusters)
    unassigned = set(groups)
    assignments: dict[str, str] = {}

    for split in ("test", "validation"):
        remaining = targets[split]
        ranked_groups = sorted(unassigned, key=lambda group: _rank_group(group, split))
        for group in ranked_groups:
            if len(group) > remaining:
                continue
            for document_id in group:
                assignments[document_id] = split
            unassigned.remove(group)
            remaining -= len(group)
            if remaining == 0:
                break

    for group in unassigned:
        for document_id in group:
            assignments[document_id] = "train"
    return assignments, targets


def _verify_assignments(
    records: list[InputRecord],
    clusters: list[dict[str, Any]],
    assignments: Mapping[str, str],
) -> None:
    document_ids = {record.document.document_id for record in records}
    if set(assignments) != document_ids:
        raise SplittingError("split assignments do not cover input exactly once")
    if set(assignments.values()) - set(SPLIT_ORDER):
        raise SplittingError("split assignments contain an unknown split")
    for cluster in clusters:
        member_splits = {assignments[member] for member in cluster["members"]}
        if len(member_splits) != 1:
            raise SplittingError("duplicate cluster crosses split boundaries")


def _completed_manifest(
    output_dir: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    expected_files = {
        "assignments": output_dir / "assignments.jsonl",
        **{split: output_dir / f"{split}.jsonl" for split in SPLIT_ORDER},
    }
    if not manifest_path.exists() and not any(
        path.exists() for path in expected_files.values()
    ):
        return None
    if not manifest_path.is_file() or not all(
        path.is_file() for path in expected_files.values()
    ):
        return None
    manifest = _read_json(manifest_path)
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise SplittingError(f"existing output does not match input field: {key}")
    file_manifest = manifest.get("files")
    if not isinstance(file_manifest, dict):
        raise SplittingError("split manifest has invalid file metadata")
    for name, path in expected_files.items():
        metadata = file_manifest.get(name)
        if not isinstance(metadata, dict) or _sha256(path) != metadata.get("sha256"):
            raise SplittingError(f"split file SHA-256 mismatch: {path.name}")
    return manifest


def split_dataset(
    input_dir: str | Path,
    deduplication_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create immutable split files and a leakage-checked membership manifest."""
    input_path = Path(input_dir)
    deduplication_path = Path(deduplication_dir)
    output_path = Path(output_dir)
    records, input_identity = _load_input(input_path)
    document_ids = {record.document.document_id for record in records}
    clusters, deduplication_identity = _load_duplicate_clusters(
        deduplication_path,
        input_identity,
        document_ids,
    )
    identity = {**input_identity, **deduplication_identity}
    output_path.mkdir(parents=True, exist_ok=True)
    completed = _completed_manifest(output_path, identity)
    if completed is not None:
        return completed

    assignments, target_counts = assign_splits(
        [record.document.document_id for record in records],
        clusters,
    )
    _verify_assignments(records, clusters, assignments)

    split_records = {
        split: [
            record
            for record in records
            if assignments[record.document.document_id] == split
        ]
        for split in SPLIT_ORDER
    }
    split_paths = {split: output_path / f"{split}.jsonl" for split in SPLIT_ORDER}
    for split, path in split_paths.items():
        _write_lines_atomic(path, (record.json_line for record in split_records[split]))

    assignment_path = output_path / "assignments.jsonl"
    assignment_lines = (
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "document_id": record.document.document_id,
                "source_id": record.document.source_id,
                "split": assignments[record.document.document_id],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )
    _write_lines_atomic(assignment_path, assignment_lines)

    language_counts: dict[str, Counter[str]] = {}
    quality_counts: dict[str, Counter[str]] = {}
    privacy_counts: dict[str, Counter[str]] = {}
    for split in SPLIT_ORDER:
        language_counts[split] = Counter(
            record.document.language for record in split_records[split]
        )
        quality_counts[split] = Counter(
            warning
            for record in split_records[split]
            for warning in record.document.quality_warnings
        )
        privacy_counts[split] = Counter(
            warning
            for record in split_records[split]
            for warning in record.document.privacy_warnings
        )

    all_paths = {"assignments": assignment_path, **split_paths}
    files = {
        name: {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "record_count": len(records)
            if name == "assignments"
            else len(split_records[name]),
        }
        for name, path in all_paths.items()
    }
    split_counts = {split: len(split_records[split]) for split in SPLIT_ORDER}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "algorithm": {
            "name": "stable_group_hash_quota",
            "version": SPLIT_VERSION,
            "salt": SPLIT_SALT,
            "ratios": SPLIT_RATIOS,
            "evaluation_selection_order": ["test", "validation"],
            "train_assignment": "remaining_groups",
        },
        "frozen": True,
        "record_count": len(records),
        "target_counts": target_counts,
        "split_counts": split_counts,
        "language_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in language_counts.items()
        },
        "quality_warning_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in quality_counts.items()
        },
        "privacy_warning_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in privacy_counts.items()
        },
        "duplicate_cluster_count": len(clusters),
        "cross_split_duplicate_cluster_count": 0,
        "overlap_document_count": 0,
        "files": files,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create frozen train, validation, and test splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-clean-v1"),
    )
    parser.add_argument(
        "--deduplication-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-dedup-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-split-v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = split_dataset(
        args.input_dir,
        args.deduplication_dir,
        args.output_dir,
    )
    counts = manifest["split_counts"]
    print(
        "Wikipedia split complete: "
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"test={counts['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
