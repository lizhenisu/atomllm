"""Formal-data v0 cleaning, deduplication, and split orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.cleaning import clean_dataset
from atomllm.data.deduplication import analyze_duplicates
from atomllm.data.splitting import split_dataset


FORMAL_PROCESSING_VERSION = "formal-process-v0"
DEFAULT_ACQUIRED_DIR = Path("artifacts/data/formal-v0/acquired-v8")
DEFAULT_CLEAN_DIR = Path("artifacts/data/formal-v0/clean-v5")
DEFAULT_DEDUPLICATION_DIR = Path("artifacts/data/formal-v0/dedup-v5")
DEFAULT_SPLIT_DIR = Path("artifacts/data/formal-v0/split-v5")
DEFAULT_PROCESSING_DIR = Path("artifacts/data/formal-v0/processed-v5")


class FormalProcessingError(RuntimeError):
    """Raised when formal-data processing cannot safely continue."""


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
        raise FormalProcessingError(f"cannot read JSON file: {path}") from error
    if not isinstance(value, dict):
        raise FormalProcessingError(f"JSON file must contain an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_acquired_manifest(acquired_dir: Path) -> dict[str, Any]:
    manifest_path = acquired_dir / "manifest.json"
    documents_path = acquired_dir / "documents.jsonl"
    if not manifest_path.is_file() or not documents_path.is_file():
        raise FormalProcessingError(
            "acquired_dir must contain manifest.json and documents.jsonl"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("formal_training_eligible") is not True:
        raise FormalProcessingError("formal acquisition is not training eligible")
    documents_sha256 = manifest.get("documents_sha256")
    if not isinstance(documents_sha256, str) or len(documents_sha256) != 64:
        raise FormalProcessingError("acquired manifest has invalid documents_sha256")
    if _sha256(documents_path) != documents_sha256:
        raise FormalProcessingError("acquired documents SHA-256 mismatch")
    return manifest


def process_formal_v0(
    *,
    project_root: str | Path = ".",
    acquired_dir: str | Path = DEFAULT_ACQUIRED_DIR,
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
    deduplication_dir: str | Path = DEFAULT_DEDUPLICATION_DIR,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    processing_dir: str | Path = DEFAULT_PROCESSING_DIR,
) -> dict[str, Any]:
    """Run formal v0 clean-v1, dedup-v1, and split-v1 in order."""
    root = Path(project_root)
    acquired_path = root / acquired_dir
    clean_path = root / clean_dir
    deduplication_path = root / deduplication_dir
    split_path = root / split_dir
    processing_path = root / processing_dir
    processing_path.mkdir(parents=True, exist_ok=True)

    acquired_manifest = _validate_acquired_manifest(acquired_path)
    clean_manifest = clean_dataset(acquired_path, clean_path)
    deduplication_manifest = analyze_duplicates(clean_path, deduplication_path)
    split_manifest = split_dataset(clean_path, deduplication_path, split_path)

    manifest = {
        "schema_version": 1,
        "processing_version": FORMAL_PROCESSING_VERSION,
        "formal_training_eligible": True,
        "acquired_dir": _relative(acquired_path, root),
        "clean_dir": _relative(clean_path, root),
        "deduplication_dir": _relative(deduplication_path, root),
        "split_dir": _relative(split_path, root),
        "acquired": {
            "record_count": acquired_manifest["record_count"],
            "estimated_tokens": acquired_manifest["estimated_tokens"],
            "documents_sha256": acquired_manifest["documents_sha256"],
        },
        "cleaning": {
            "record_count": clean_manifest["record_count"],
            "changed_document_count": clean_manifest["changed_document_count"],
            "quality_warning_counts": clean_manifest["quality_warning_counts"],
            "privacy_warning_counts": clean_manifest["privacy_warning_counts"],
            "documents_sha256": clean_manifest["documents_sha256"],
        },
        "deduplication": {
            "record_count": deduplication_manifest["record_count"],
            "exact_cluster_count": deduplication_manifest["exact_cluster_count"],
            "near_cluster_count": deduplication_manifest["near_cluster_count"],
            "exact_duplicate_document_count": deduplication_manifest[
                "exact_duplicate_document_count"
            ],
            "near_duplicate_pair_count": deduplication_manifest[
                "near_duplicate_pair_count"
            ],
            "clusters_sha256": deduplication_manifest["clusters_sha256"],
        },
        "splitting": {
            "record_count": split_manifest["record_count"],
            "split_counts": split_manifest["split_counts"],
            "duplicate_cluster_count": split_manifest["duplicate_cluster_count"],
            "cross_split_duplicate_cluster_count": split_manifest[
                "cross_split_duplicate_cluster_count"
            ],
            "overlap_document_count": split_manifest["overlap_document_count"],
            "files": split_manifest["files"],
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(processing_path / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process formal-data v0 through cleaning, deduplication, and splits."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--acquired-dir", type=Path, default=DEFAULT_ACQUIRED_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument(
        "--deduplication-dir",
        type=Path,
        default=DEFAULT_DEDUPLICATION_DIR,
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--processing-dir",
        type=Path,
        default=DEFAULT_PROCESSING_DIR,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = process_formal_v0(
        project_root=args.project_root,
        acquired_dir=args.acquired_dir,
        clean_dir=args.clean_dir,
        deduplication_dir=args.deduplication_dir,
        split_dir=args.split_dir,
        processing_dir=args.processing_dir,
    )
    print(
        json.dumps(
            {
                "cleaned_records": manifest["cleaning"]["record_count"],
                "dedup_exact_clusters": manifest["deduplication"][
                    "exact_cluster_count"
                ],
                "dedup_near_clusters": manifest["deduplication"]["near_cluster_count"],
                "formal_training_eligible": manifest["formal_training_eligible"],
                "split_counts": manifest["splitting"]["split_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
