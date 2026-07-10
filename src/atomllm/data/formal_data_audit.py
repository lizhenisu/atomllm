"""End-to-end audit for a formal train/validation data version."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.formal_exact_dedup import _read_json, _sha256, _write_json_atomic
from atomllm.data.formal_acquisition import estimate_tokens
from atomllm.data.formal_split_config import FormalSplitConfig, load_formal_split_config
from atomllm.data.schema import CanonicalDocument


AUDIT_VERSION = "formal-70g-audit-v1"
DEFAULT_ACQUISITION_DIR = Path("artifacts/data/formal-70g/acquired-space-v1")
DEFAULT_CLEAN_DIR = Path("artifacts/data/formal-70g/clean-v1")
DEFAULT_EXACT_DIR = Path("artifacts/data/formal-70g/exact-dedup-v1")
DEFAULT_NEAR_DIR = Path("artifacts/data/formal-70g/near-dedup-v8")
DEFAULT_SPLIT_DIR = Path("artifacts/data/formal-70g/split-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/data/formal-70g/audit-v1")
DEFAULT_CONFIG = Path("configs/data/formal-70g-processing.yaml")


class FormalAuditError(RuntimeError):
    """Raised when formal 70G audit evidence is missing or inconsistent."""


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def _verify_split_shards(
    split_dir: Path, split_manifest: dict[str, Any], split_config: FormalSplitConfig
) -> tuple[dict[str, int], Counter[str], Counter[str], set[str]]:
    raw = split_manifest.get("shards")
    if not isinstance(raw, dict) or set(raw) != {"train", "validation"}:
        raise FormalAuditError("split manifest must contain only train and validation")
    counts: dict[str, int] = {}
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    validation_ids: set[str] = set()
    for split in ("validation", "train"):
        metadata = raw[split]
        if not isinstance(metadata, list) or not metadata:
            raise FormalAuditError(f"split shard metadata is invalid: {split}")
        total = 0
        for item in metadata:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise FormalAuditError(f"invalid shard metadata: {split}")
            path = split_dir / split / "shards" / item["name"]
            digest, line_count = _sha256_and_lines(path)
            if digest != item.get("sha256") or line_count != item.get("record_count"):
                raise FormalAuditError(f"split shard integrity mismatch: {path}")
            total += line_count
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    document = CanonicalDocument.from_json_line(line)
                    languages[document.language] += 1
                    sources[document.source_id] += 1
                    if split == "validation":
                        if document.document_id in validation_ids:
                            raise FormalAuditError("duplicate document in validation")
                        validation_ids.add(document.document_id)
                        if document.quality_warnings:
                            raise FormalAuditError("validation contains quality warnings")
                        tokens = estimate_tokens(document.text, document.language)
                        if not (
                            split_config.min_estimated_tokens
                            <= tokens
                            <= split_config.max_estimated_tokens
                        ):
                            raise FormalAuditError(
                                "validation document violates token-length window"
                            )
                    elif document.document_id in validation_ids:
                        raise FormalAuditError("train/validation document overlap")
        counts[split] = total
    return counts, languages, sources, validation_ids


def audit_formal_data(
    *,
    project_root: str | Path = ".",
    acquisition_dir: str | Path = DEFAULT_ACQUISITION_DIR,
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
    exact_dir: str | Path = DEFAULT_EXACT_DIR,
    near_dir: str | Path = DEFAULT_NEAR_DIR,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Verify final data integrity, proportions and trainability contracts."""
    root = Path(project_root)
    acquisition_path = root / acquisition_dir
    clean_path = root / clean_dir
    exact_path = root / exact_dir
    near_path = root / near_dir
    split_path = root / split_dir
    split_config = load_formal_split_config(root / config_path)
    manifests = {
        "acquisition": _read_json(acquisition_path / "manifest.json"),
        "clean": _read_json(clean_path / "manifest.json"),
        "exact": _read_json(exact_path / "manifest.json"),
        "near": _read_json(near_path / "manifest.json"),
        "split": _read_json(split_path / "manifest.json"),
    }
    if not manifests["acquisition"].get("formal_training_eligible"):
        raise FormalAuditError("acquisition is not training eligible")
    if manifests["clean"].get("privacy_action") != "warn":
        raise FormalAuditError("privacy action must remain warn")
    if manifests["clean"].get("quality_action") != "warn":
        raise FormalAuditError("quality action must remain warn")
    for name, path, manifest in (
        ("exact", exact_path, manifests["exact"]),
        ("near", near_path, manifests["near"]),
    ):
        database = path / str(manifest.get("database_file", ""))
        if not database.is_file() or _sha256(database) != manifest.get("database_sha256"):
            raise FormalAuditError(f"{name} database SHA-256 mismatch")
    if _sha256(near_path / str(manifests["near"].get("clusters_file", ""))) != manifests[
        "near"
    ].get("clusters_sha256"):
        raise FormalAuditError("near cluster SHA-256 mismatch")

    expected = {
        "clean_manifest_sha256": _sha256(clean_path / "manifest.json"),
        "exact_manifest_sha256": _sha256(exact_path / "manifest.json"),
        "near_manifest_sha256": _sha256(near_path / "manifest.json"),
    }
    if any(manifests["split"].get(key) != value for key, value in expected.items()):
        raise FormalAuditError("split provenance manifest mismatch")
    language_tokens = manifests["acquisition"].get("estimated_tokens_by_language_bucket")
    if not isinstance(language_tokens, dict):
        raise FormalAuditError("acquisition language tokens are missing")
    ordered = ["zh-Hans", "en", "zh-Hant", "ja", "other"]
    if any(
        language_tokens[left] <= language_tokens[right]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise FormalAuditError("acquisition language order contract failed")
    fractions = manifests["acquisition"].get("actual_source_document_fractions")
    if not isinstance(fractions, dict) or any(value > 0.2 for value in fractions.values()):
        raise FormalAuditError("single-source fraction contract failed")
    counts, languages, sources, validation_ids = _verify_split_shards(
        split_path, manifests["split"], split_config
    )
    declared = manifests["split"].get("splits")
    if counts != declared or sum(counts.values()) != manifests["split"].get(
        "retained_after_dedup_count"
    ):
        raise FormalAuditError("split count coverage mismatch")
    if counts["validation"] != round(
        sum(counts.values()) * split_config.validation_fraction
    ):
        raise FormalAuditError("validation count does not match split config")
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    audit = {
        "audit_version": AUDIT_VERSION,
        "training_eligible": True,
        "checks": {
            "input_provenance": True,
            "privacy_warning_mode": True,
            "quality_warning_mode": True,
            "dedup_artifact_hashes": True,
            "split_shard_hashes": True,
            "split_coverage": True,
            "train_validation_disjoint": True,
            "validation_quality_and_token_window": True,
            "language_order": True,
            "single_source_fraction": True,
        },
        "counts": counts,
        "validation_document_ids": len(validation_ids),
        "output_language_counts": dict(sorted(languages.items())),
        "output_source_counts": dict(sorted(sources.items())),
        "provenance": {key: _sha256(path / "manifest.json") for key, path in (
            ("acquisition", acquisition_path),
            ("clean", clean_path),
            ("exact", exact_path),
            ("near", near_path),
            ("split", split_path),
        )},
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(output_path / "manifest.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit formal train/validation data.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    audit = audit_formal_data(project_root=args.project_root, config_path=args.config)
    print(json.dumps({"training_eligible": audit["training_eligible"], "counts": audit["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
