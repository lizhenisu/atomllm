"""Deterministic exact deduplication and conservative near-duplicate detection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.schema import SCHEMA_VERSION, CanonicalDocument


DEDUPLICATION_VERSION = "dedup-v1"
SHINGLE_CHARS = 24
SHINGLE_SAMPLE_BITS = 4
SIMHASH_BITS = 64
SIMHASH_BANDS = 8
NEAR_DUPLICATE_THRESHOLD = 0.90
MIN_LENGTH_RATIO = 0.90

_WHITESPACE = re.compile(r"\s+")
_ROLLING_HASH_BASE = 1_000_003
_UINT64_MASK = (1 << 64) - 1


class DeduplicationError(RuntimeError):
    """Raised when a deduplication analysis cannot safely continue."""


@dataclass(frozen=True, slots=True)
class DocumentFingerprint:
    document_id: str
    input_index: int
    character_count: int
    text_sha256: str
    shingles: frozenset[int]
    simhash: int


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
        raise DeduplicationError(f"cannot read JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise DeduplicationError(f"JSON file is not an object: {path.name}")
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


def _write_jsonl_atomic(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            line = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(f"{line}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _hash_shingle(shingle: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            shingle.encode("utf-8"),
            digest_size=8,
            person=b"atomllm-dedup-v1",
        ).digest(),
        "big",
    )


def _character_shingles(text: str) -> frozenset[int]:
    compact = _WHITESPACE.sub("", text)
    if len(compact) <= SHINGLE_CHARS:
        return frozenset({_hash_shingle(compact)})

    window_hash = 0
    for character in compact[:SHINGLE_CHARS]:
        window_hash = (window_hash * _ROLLING_HASH_BASE + ord(character)) & _UINT64_MASK
    sampled = (
        {window_hash} if window_hash & ((1 << SHINGLE_SAMPLE_BITS) - 1) == 0 else set()
    )
    leading_power = pow(
        _ROLLING_HASH_BASE,
        SHINGLE_CHARS - 1,
        _UINT64_MASK + 1,
    )
    for start in range(1, len(compact) - SHINGLE_CHARS + 1):
        outgoing = ord(compact[start - 1])
        incoming = ord(compact[start + SHINGLE_CHARS - 1])
        window_hash = (
            (window_hash - outgoing * leading_power) * _ROLLING_HASH_BASE + incoming
        ) & _UINT64_MASK
        if window_hash & ((1 << SHINGLE_SAMPLE_BITS) - 1) == 0:
            sampled.add(window_hash)
    if not sampled:
        sampled.add(_hash_shingle(compact))
    return frozenset(sampled)


def _simhash(shingles: frozenset[int]) -> int:
    bit_weights = [0] * SIMHASH_BITS
    for value in shingles:
        for bit in range(SIMHASH_BITS):
            bit_weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(bit_weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def fingerprint_document(
    document: CanonicalDocument, input_index: int
) -> DocumentFingerprint:
    shingles = _character_shingles(document.text)
    return DocumentFingerprint(
        document_id=document.document_id,
        input_index=input_index,
        character_count=len(document.text),
        text_sha256=hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        shingles=shingles,
        simhash=_simhash(shingles),
    )


def _cluster_id(kind: str, members: Iterable[str]) -> str:
    identity = f"{kind}\0{chr(0).join(sorted(members))}".encode()
    return f"cluster-{hashlib.sha256(identity).hexdigest()}"


def _exact_clusters(
    fingerprints: list[DocumentFingerprint],
) -> tuple[list[dict[str, Any]], list[DocumentFingerprint]]:
    by_hash: dict[str, list[DocumentFingerprint]] = defaultdict(list)
    for fingerprint in fingerprints:
        by_hash[fingerprint.text_sha256].append(fingerprint)

    clusters: list[dict[str, Any]] = []
    representatives: list[DocumentFingerprint] = []
    for group in sorted(by_hash.values(), key=lambda items: items[0].input_index):
        representative = min(group, key=lambda item: item.input_index)
        representatives.append(representative)
        if len(group) < 2:
            continue
        members = [
            item.document_id
            for item in sorted(group, key=lambda item: item.input_index)
        ]
        clusters.append(
            {
                "cluster_id": _cluster_id("exact", members),
                "kind": "exact",
                "representative_document_id": representative.document_id,
                "members": members,
                "text_sha256": representative.text_sha256,
            }
        )
    return clusters, representatives


def _candidate_pairs(
    fingerprints: list[DocumentFingerprint],
) -> set[tuple[int, int]]:
    band_bits = SIMHASH_BITS // SIMHASH_BANDS
    band_mask = (1 << band_bits) - 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for index, fingerprint in enumerate(fingerprints):
        for band in range(SIMHASH_BANDS):
            value = (fingerprint.simhash >> (band * band_bits)) & band_mask
            key = (band, value)
            for other_index in buckets[key]:
                candidates.add((other_index, index))
            buckets[key].append(index)
    return candidates


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 1.0


def _near_clusters(
    fingerprints: list[DocumentFingerprint],
) -> tuple[list[dict[str, Any]], int, int]:
    candidates = _candidate_pairs(fingerprints)
    edges: list[tuple[str, str, float]] = []
    disjoint_set = _DisjointSet(fingerprint.document_id for fingerprint in fingerprints)
    for left_index, right_index in sorted(candidates):
        left = fingerprints[left_index]
        right = fingerprints[right_index]
        length_ratio = min(left.character_count, right.character_count) / max(
            left.character_count, right.character_count
        )
        if length_ratio < MIN_LENGTH_RATIO:
            continue
        similarity = _jaccard(left.shingles, right.shingles)
        if similarity < NEAR_DUPLICATE_THRESHOLD:
            continue
        disjoint_set.union(left.document_id, right.document_id)
        edges.append((left.document_id, right.document_id, similarity))

    by_root: dict[str, list[DocumentFingerprint]] = defaultdict(list)
    for fingerprint in fingerprints:
        by_root[disjoint_set.find(fingerprint.document_id)].append(fingerprint)

    clusters: list[dict[str, Any]] = []
    for group in sorted(by_root.values(), key=lambda items: items[0].input_index):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: item.input_index)
        member_ids = [item.document_id for item in ordered]
        member_set = set(member_ids)
        similarities = [
            {
                "left_document_id": left,
                "right_document_id": right,
                "similarity": round(similarity, 6),
            }
            for left, right, similarity in edges
            if left in member_set and right in member_set
        ]
        clusters.append(
            {
                "cluster_id": _cluster_id("near", member_ids),
                "kind": "near",
                "representative_document_id": ordered[0].document_id,
                "members": member_ids,
                "similarity_edges": similarities,
            }
        )
    return clusters, len(candidates), len(edges)


def _validate_input(input_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path = input_dir / "manifest.json"
    documents_path = input_dir / "documents.jsonl"
    if not manifest_path.is_file() or not documents_path.is_file():
        raise DeduplicationError(
            "input directory must contain manifest.json and documents.jsonl"
        )
    manifest = _read_json(manifest_path)
    record_count = manifest.get("record_count")
    documents_sha256 = manifest.get("documents_sha256")
    if type(record_count) is not int or record_count <= 0:
        raise DeduplicationError("input manifest has an invalid record_count")
    if not isinstance(documents_sha256, str) or len(documents_sha256) != 64:
        raise DeduplicationError("input manifest has an invalid documents_sha256")
    if _sha256(documents_path) != documents_sha256:
        raise DeduplicationError("input documents SHA-256 does not match its manifest")
    with documents_path.open("rb") as handle:
        if sum(1 for _ in handle) != record_count:
            raise DeduplicationError(
                "input documents line count does not match its manifest"
            )
    identity = {
        "deduplication_version": DEDUPLICATION_VERSION,
        "input_record_count": record_count,
        "input_documents_sha256": documents_sha256,
        "input_manifest_sha256": _sha256(manifest_path),
    }
    return documents_path, manifest, identity


def _completed_manifest(
    output_dir: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    clusters_path = output_dir / "duplicate-clusters.jsonl"
    if not manifest_path.exists() and not clusters_path.exists():
        return None
    if not manifest_path.is_file() or not clusters_path.is_file():
        raise DeduplicationError("deduplication output is incomplete")
    manifest = _read_json(manifest_path)
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise DeduplicationError(
                f"existing output does not match input field: {key}"
            )
    if _sha256(clusters_path) != manifest.get("clusters_sha256"):
        raise DeduplicationError("duplicate clusters SHA-256 does not match manifest")
    return manifest


def analyze_duplicates(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Analyze exact and near duplicates without deleting any documents."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    documents_path, input_manifest, identity = _validate_input(input_path)
    output_path.mkdir(parents=True, exist_ok=True)
    completed = _completed_manifest(output_path, identity)
    if completed is not None:
        return completed

    fingerprints: list[DocumentFingerprint] = []
    seen_document_ids: set[str] = set()
    with documents_path.open(encoding="utf-8") as handle:
        for input_index, line in enumerate(handle):
            document = CanonicalDocument.from_json_line(line)
            if document.document_id in seen_document_ids:
                raise DeduplicationError("input contains a repeated document_id")
            seen_document_ids.add(document.document_id)
            fingerprints.append(fingerprint_document(document, input_index))

    exact_clusters, representatives = _exact_clusters(fingerprints)
    near_clusters, candidate_pair_count, near_pair_count = _near_clusters(
        representatives
    )
    clusters = [*exact_clusters, *near_clusters]
    clusters_path = output_path / "duplicate-clusters.jsonl"
    _write_jsonl_atomic(clusters_path, clusters)

    exact_duplicate_documents = sum(
        len(cluster["members"]) - 1 for cluster in exact_clusters
    )
    near_candidate_documents = len(
        {document_id for cluster in near_clusters for document_id in cluster["members"]}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "algorithm": {
            "name": "exact_sha256_plus_simhash_jaccard",
            "version": DEDUPLICATION_VERSION,
            "shingle_chars": SHINGLE_CHARS,
            "shingle_sample_bits": SHINGLE_SAMPLE_BITS,
            "simhash_bits": SIMHASH_BITS,
            "simhash_bands": SIMHASH_BANDS,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "minimum_length_ratio": MIN_LENGTH_RATIO,
        },
        "source_id": input_manifest.get("source_id"),
        "selection_rule": "earliest_input_document",
        "action": "report_only",
        "record_count": len(fingerprints),
        "retained_count": len(fingerprints),
        "dropped_count": 0,
        "unique_text_count": len(representatives),
        "exact_cluster_count": len(exact_clusters),
        "exact_duplicate_document_count": exact_duplicate_documents,
        "near_cluster_count": len(near_clusters),
        "near_candidate_document_count": near_candidate_documents,
        "lsh_candidate_pair_count": candidate_pair_count,
        "near_duplicate_pair_count": near_pair_count,
        "clusters_file": clusters_path.name,
        "clusters_bytes": clusters_path.stat().st_size,
        "clusters_sha256": _sha256(clusters_path),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze exact and near duplicates in the cleaned Wikipedia sample."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-clean-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-dedup-v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = analyze_duplicates(args.input_dir, args.output_dir)
    print(
        "Wikipedia deduplication analysis complete: "
        f"{manifest['exact_cluster_count']} exact clusters, "
        f"{manifest['near_cluster_count']} near clusters, "
        f"{manifest['dropped_count']} documents dropped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
