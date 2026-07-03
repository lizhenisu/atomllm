"""Build a deterministic, machine-verifiable data version from pipeline manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atomllm.data.schema import SCHEMA_VERSION


DATA_VERSION_SCHEMA_VERSION = 1
DATA_VERSION_NAME = "wikipedia-20231101-zh-smoke-v1"


class DataVersionError(RuntimeError):
    """Raised when pipeline lineage is incomplete, inconsistent, or modified."""


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
        raise DataVersionError(f"cannot read JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise DataVersionError(f"JSON file is not an object: {path.name}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise DataVersionError(message)


def _load_stage(directory: Path, stage: str) -> tuple[dict[str, Any], Path, str]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise DataVersionError(f"{stage} manifest.json is missing")
    manifest = _read_json(manifest_path)
    return manifest, manifest_path, _sha256(manifest_path)


def _validate_document_stage(
    directory: Path,
    manifest: Mapping[str, Any],
    stage: str,
) -> tuple[Path, str, int]:
    documents_path = directory / "documents.jsonl"
    if not documents_path.is_file():
        raise DataVersionError(f"{stage} documents.jsonl is missing")
    expected_sha256 = manifest.get("documents_sha256")
    record_count = manifest.get("record_count")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise DataVersionError(f"{stage} manifest has invalid documents_sha256")
    if type(record_count) is not int or record_count <= 0:
        raise DataVersionError(f"{stage} manifest has invalid record_count")
    actual_sha256 = _sha256(documents_path)
    _require_equal(
        actual_sha256,
        expected_sha256,
        f"{stage} documents SHA-256 does not match manifest",
    )
    with documents_path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    _require_equal(
        line_count,
        record_count,
        f"{stage} documents line count does not match manifest",
    )
    return documents_path, actual_sha256, record_count


def _validate_split_files(
    directory: Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DataVersionError("splitting manifest has invalid files metadata")
    verified: dict[str, dict[str, Any]] = {}
    for name in ("assignments", "train", "validation", "test"):
        metadata = files.get(name)
        if not isinstance(metadata, dict):
            raise DataVersionError(
                f"splitting manifest is missing file metadata: {name}"
            )
        filename = metadata.get("name")
        expected_sha256 = metadata.get("sha256")
        record_count = metadata.get("record_count")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or type(record_count) is not int
            or record_count < 0
        ):
            raise DataVersionError(f"splitting file metadata is invalid: {name}")
        path = directory / filename
        if not path.is_file():
            raise DataVersionError(f"splitting output file is missing: {filename}")
        _require_equal(
            _sha256(path),
            expected_sha256,
            f"splitting output SHA-256 does not match: {filename}",
        )
        with path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        _require_equal(
            line_count,
            record_count,
            f"splitting output line count does not match: {filename}",
        )
        verified[name] = {
            "name": filename,
            "record_count": record_count,
            "sha256": expected_sha256,
        }
    return verified


def _build_payload(
    acquisition_dir: Path,
    cleaning_dir: Path,
    deduplication_dir: Path,
    splitting_dir: Path,
) -> dict[str, Any]:
    acquisition, acquisition_manifest_path, acquisition_manifest_sha256 = _load_stage(
        acquisition_dir, "acquisition"
    )
    _, acquisition_documents_sha256, acquisition_count = _validate_document_stage(
        acquisition_dir,
        acquisition,
        "acquisition",
    )
    _require_equal(
        acquisition.get("source_enabled"),
        False,
        "smoke data source must remain disabled",
    )
    _require_equal(
        acquisition.get("purpose"),
        "acquisition_smoke",
        "acquisition purpose must be acquisition_smoke",
    )

    cleaning, cleaning_manifest_path, cleaning_manifest_sha256 = _load_stage(
        cleaning_dir,
        "cleaning",
    )
    _, cleaning_documents_sha256, cleaning_count = _validate_document_stage(
        cleaning_dir,
        cleaning,
        "cleaning",
    )
    _require_equal(
        cleaning.get("input_documents_sha256"),
        acquisition_documents_sha256,
        "cleaning input documents do not match acquisition output",
    )
    _require_equal(
        cleaning.get("input_manifest_sha256"),
        acquisition_manifest_sha256,
        "cleaning input manifest does not match acquisition manifest",
    )
    _require_equal(
        cleaning_count,
        acquisition_count,
        "cleaning record count does not match acquisition",
    )
    _require_equal(cleaning.get("dropped_count"), 0, "cleaning must not drop records")

    deduplication, deduplication_manifest_path, deduplication_manifest_sha256 = (
        _load_stage(deduplication_dir, "deduplication")
    )
    clusters_path = deduplication_dir / str(
        deduplication.get("clusters_file", "duplicate-clusters.jsonl")
    )
    if not clusters_path.is_file():
        raise DataVersionError("deduplication cluster file is missing")
    clusters_sha256 = _sha256(clusters_path)
    _require_equal(
        clusters_sha256,
        deduplication.get("clusters_sha256"),
        "deduplication clusters SHA-256 does not match manifest",
    )
    _require_equal(
        deduplication.get("input_documents_sha256"),
        cleaning_documents_sha256,
        "deduplication input documents do not match cleaning output",
    )
    _require_equal(
        deduplication.get("input_manifest_sha256"),
        cleaning_manifest_sha256,
        "deduplication input manifest does not match cleaning manifest",
    )
    _require_equal(
        deduplication.get("record_count"),
        cleaning_count,
        "deduplication record count does not match cleaning",
    )
    _require_equal(
        deduplication.get("action"),
        "report_only",
        "deduplication action must remain report_only",
    )
    _require_equal(
        deduplication.get("dropped_count"),
        0,
        "deduplication must not drop records",
    )

    splitting, splitting_manifest_path, splitting_manifest_sha256 = _load_stage(
        splitting_dir,
        "splitting",
    )
    _require_equal(
        splitting.get("input_documents_sha256"),
        cleaning_documents_sha256,
        "splitting input documents do not match cleaning output",
    )
    _require_equal(
        splitting.get("input_manifest_sha256"),
        cleaning_manifest_sha256,
        "splitting input manifest does not match cleaning manifest",
    )
    _require_equal(
        splitting.get("deduplication_manifest_sha256"),
        deduplication_manifest_sha256,
        "splitting does not reference the verified deduplication manifest",
    )
    _require_equal(
        splitting.get("duplicate_clusters_sha256"),
        clusters_sha256,
        "splitting does not reference the verified duplicate clusters",
    )
    _require_equal(splitting.get("frozen"), True, "splitting manifest is not frozen")
    _require_equal(
        splitting.get("record_count"),
        cleaning_count,
        "splitting record count does not match cleaning",
    )
    _require_equal(
        splitting.get("overlap_document_count"),
        0,
        "splitting contains overlapping documents",
    )
    _require_equal(
        splitting.get("cross_split_duplicate_cluster_count"),
        0,
        "splitting leaks duplicate clusters across sets",
    )
    split_counts = splitting.get("split_counts")
    if not isinstance(split_counts, dict):
        raise DataVersionError("splitting manifest has invalid split_counts")
    _require_equal(
        sum(split_counts.values()),
        cleaning_count,
        "split counts do not cover all cleaned records",
    )
    split_files = _validate_split_files(splitting_dir, splitting)

    return {
        "schema_version": SCHEMA_VERSION,
        "data_version_schema_version": DATA_VERSION_SCHEMA_VERSION,
        "name": DATA_VERSION_NAME,
        "purpose": "end_to_end_data_pipeline_smoke",
        "status": "smoke_validated",
        "training_eligible": False,
        "training_ineligible_reasons": [
            "source_is_disabled",
            "sample_is_limited_to_1000_documents",
            "single_source_is_not_the_pretraining_mixture",
        ],
        "source": {
            "source_id": acquisition.get("source_id"),
            "dataset": acquisition.get("dataset"),
            "revision": acquisition.get("revision"),
            "config_name": acquisition.get("config_name"),
            "split": acquisition.get("split"),
            "source_enabled": acquisition.get("source_enabled"),
        },
        "lineage": {
            "acquisition": {
                "manifest_sha256": acquisition_manifest_sha256,
                "documents_sha256": acquisition_documents_sha256,
                "record_count": acquisition_count,
            },
            "cleaning": {
                "manifest_sha256": cleaning_manifest_sha256,
                "documents_sha256": cleaning_documents_sha256,
                "record_count": cleaning_count,
                "dropped_count": cleaning.get("dropped_count"),
            },
            "deduplication": {
                "manifest_sha256": deduplication_manifest_sha256,
                "clusters_sha256": clusters_sha256,
                "record_count": deduplication.get("record_count"),
                "exact_cluster_count": deduplication.get("exact_cluster_count"),
                "near_cluster_count": deduplication.get("near_cluster_count"),
                "dropped_count": deduplication.get("dropped_count"),
            },
            "splitting": {
                "manifest_sha256": splitting_manifest_sha256,
                "record_count": splitting.get("record_count"),
                "split_counts": split_counts,
                "files": split_files,
            },
        },
        "audit": {
            "immutable_source_revision": True,
            "all_document_hashes_verified": True,
            "all_manifest_links_verified": True,
            "record_count_preserved": True,
            "privacy_action_is_warn": cleaning.get("transform", {}).get(
                "privacy_action"
            )
            == "warn",
            "quality_action_is_warn": cleaning.get("transform", {}).get(
                "quality_action"
            )
            == "warn",
            "no_automatic_drops": True,
            "split_files_verified": True,
            "split_overlap_document_count": 0,
            "cross_split_duplicate_cluster_count": 0,
        },
    }


def build_data_version(
    acquisition_dir: str | Path,
    cleaning_dir: str | Path,
    deduplication_dir: str | Path,
    splitting_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate the complete lineage and write a deterministic version manifest."""
    payload = _build_payload(
        Path(acquisition_dir),
        Path(cleaning_dir),
        Path(deduplication_dir),
        Path(splitting_dir),
    )
    identity_json = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    identity_digest = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    manifest = {
        **payload,
        "data_version_id": f"data-{DATA_VERSION_NAME}-{identity_digest[:12]}",
        "identity_sha256": identity_digest,
    }
    serialized = f"{_canonical_json(manifest)}\n"
    manifest_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    checksum_path = output_path / "manifest.sha256"
    if manifest_path.exists() or checksum_path.exists():
        if not manifest_path.is_file() or not checksum_path.is_file():
            raise DataVersionError("existing data version output is incomplete")
        if manifest_path.read_text(encoding="utf-8") != serialized:
            raise DataVersionError(
                "existing data version manifest has different content"
            )
        expected_checksum = f"{manifest_digest}  manifest.json\n"
        if checksum_path.read_text(encoding="utf-8") != expected_checksum:
            raise DataVersionError("existing data version checksum is invalid")
        return manifest

    _write_text_atomic(manifest_path, serialized)
    _write_text_atomic(checksum_path, f"{manifest_digest}  manifest.json\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the complete Wikipedia smoke data lineage."
    )
    parser.add_argument(
        "--acquisition-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-smoke"),
    )
    parser.add_argument(
        "--cleaning-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-clean-v1"),
    )
    parser.add_argument(
        "--deduplication-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-dedup-v1"),
    )
    parser.add_argument(
        "--splitting-dir",
        type=Path,
        default=Path("data/processed/wikipedia-20231101-zh-split-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data-versions/wikipedia-20231101-zh-smoke-v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_data_version(
        args.acquisition_dir,
        args.cleaning_dir,
        args.deduplication_dir,
        args.splitting_dir,
        args.output_dir,
    )
    print(
        "Data version complete: "
        f"{manifest['data_version_id']}, "
        f"status={manifest['status']}, "
        f"training_eligible={str(manifest['training_eligible']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
