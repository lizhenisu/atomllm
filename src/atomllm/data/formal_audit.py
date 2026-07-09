"""Formal-data v0 lineage and release-gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.data.formal_plan import load_formal_data_plan
from atomllm.data.mixture import load_pretraining_mixture


FORMAL_AUDIT_VERSION = "formal-audit-v0"
DEFAULT_PLAN_PATH = Path("configs/data/formal-v0-sampling-plan.yaml")
DEFAULT_MIXTURE_PATH = Path("configs/data/pretraining-mixture.yaml")
DEFAULT_ACQUIRED_DIR = Path("artifacts/data/formal-v0/acquired-v8")
DEFAULT_CLEAN_DIR = Path("artifacts/data/formal-v0/clean-v5")
DEFAULT_DEDUPLICATION_DIR = Path("artifacts/data/formal-v0/dedup-v5")
DEFAULT_SPLIT_DIR = Path("artifacts/data/formal-v0/split-v5")
DEFAULT_PROCESSING_DIR = Path("artifacts/data/formal-v0/processed-v5")
DEFAULT_AUDIT_DIR = Path("artifacts/data/formal-v0/audit-v4")
LANGUAGE_PRIORITY = ("zh-Hans", "en", "zh-Hant", "ja", "other")


class FormalAuditError(RuntimeError):
    """Raised when formal-data v0 audit inputs cannot be read safely."""


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
        raise FormalAuditError(f"cannot read JSON file: {path}") from error
    if not isinstance(value, dict):
        raise FormalAuditError(f"JSON file must contain an object: {path}")
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


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _manifest(directory: Path, stage: str) -> tuple[dict[str, Any], str]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FormalAuditError(f"{stage} manifest.json is missing")
    return _read_json(path), _sha256(path)


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def _verify_documents(
    directory: Path,
    manifest: Mapping[str, Any],
    stage: str,
    checks: list[dict[str, Any]],
) -> None:
    documents_file = manifest.get("documents_file", "documents.jsonl")
    if not isinstance(documents_file, str):
        raise FormalAuditError(f"{stage} documents_file is invalid")
    path = directory / documents_file
    if not path.is_file():
        raise FormalAuditError(f"{stage} documents file is missing")
    expected_sha256 = manifest.get("documents_sha256")
    record_count = manifest.get("record_count")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise FormalAuditError(f"{stage} documents_sha256 is invalid")
    if type(record_count) is not int or record_count <= 0:
        raise FormalAuditError(f"{stage} record_count is invalid")
    actual_sha256 = _sha256(path)
    actual_lines = _count_lines(path)
    _record(
        checks,
        f"{stage}_documents_sha256",
        actual_sha256 == expected_sha256,
        f"{actual_sha256}",
    )
    _record(
        checks,
        f"{stage}_record_count",
        actual_lines == record_count,
        f"{actual_lines} lines, manifest={record_count}",
    )


def _verify_split_files(
    split_dir: Path,
    split: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    files = split.get("files")
    if not isinstance(files, dict):
        raise FormalAuditError("split manifest has invalid files metadata")
    for name in ("assignments", "train", "validation", "test"):
        metadata = files.get(name)
        if not isinstance(metadata, dict):
            raise FormalAuditError(f"split file metadata is missing: {name}")
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
            raise FormalAuditError(f"split file metadata is invalid: {name}")
        path = split_dir / filename
        if not path.is_file():
            raise FormalAuditError(f"split output file is missing: {filename}")
        actual_sha256 = _sha256(path)
        actual_lines = _count_lines(path)
        _record(
            checks,
            f"split_{name}_sha256",
            actual_sha256 == expected_sha256,
            f"{actual_sha256}",
        )
        _record(
            checks,
            f"split_{name}_record_count",
            actual_lines == record_count,
            f"{actual_lines} lines, manifest={record_count}",
        )


def _fraction_map(values: Mapping[str, Any], total: int) -> dict[str, float]:
    return {
        key: round(value / total, 6)
        for key, value in sorted(values.items())
        if isinstance(value, int)
    }


def audit_formal_v0(
    *,
    project_root: str | Path = ".",
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    mixture_path: str | Path = DEFAULT_MIXTURE_PATH,
    acquired_dir: str | Path = DEFAULT_ACQUIRED_DIR,
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
    deduplication_dir: str | Path = DEFAULT_DEDUPLICATION_DIR,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    processing_dir: str | Path = DEFAULT_PROCESSING_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict[str, Any]:
    """Audit formal v0 artifacts and write a release-gate manifest."""
    root = Path(project_root)
    plan = load_formal_data_plan(root / plan_path, project_root=root)
    mixture = load_pretraining_mixture(root / mixture_path)
    acquired_path = root / acquired_dir
    clean_path = root / clean_dir
    deduplication_path = root / deduplication_dir
    split_path = root / split_dir
    processing_path = root / processing_dir
    audit_path = root / audit_dir
    audit_path.mkdir(parents=True, exist_ok=True)

    acquired, acquired_manifest_sha256 = _manifest(acquired_path, "acquisition")
    clean, clean_manifest_sha256 = _manifest(clean_path, "cleaning")
    deduplication, deduplication_manifest_sha256 = _manifest(
        deduplication_path, "deduplication"
    )
    split, split_manifest_sha256 = _manifest(split_path, "splitting")
    processing, processing_manifest_sha256 = _manifest(processing_path, "processing")

    checks: list[dict[str, Any]] = []
    _verify_documents(acquired_path, acquired, "acquisition", checks)
    _verify_documents(clean_path, clean, "cleaning", checks)
    _verify_split_files(split_path, split, checks)

    clusters_path = deduplication_path / str(
        deduplication.get("clusters_file", "duplicate-clusters.jsonl")
    )
    if not clusters_path.is_file():
        raise FormalAuditError("deduplication clusters file is missing")
    clusters_sha256 = _sha256(clusters_path)
    _record(
        checks,
        "deduplication_clusters_sha256",
        clusters_sha256 == deduplication.get("clusters_sha256"),
        clusters_sha256,
    )

    acquired_documents_sha256 = acquired["documents_sha256"]
    clean_documents_sha256 = clean["documents_sha256"]
    _record(
        checks,
        "acquisition_training_eligible",
        acquired.get("formal_training_eligible") is True,
        str(acquired.get("formal_training_eligible")),
    )
    _record(
        checks,
        "formal_plan_training_eligible",
        plan.training_eligible is True,
        str(plan.training_eligible),
    )
    _record(
        checks,
        "processing_training_eligible",
        processing.get("formal_training_eligible") is True,
        str(processing.get("formal_training_eligible")),
    )
    _record(
        checks,
        "cleaning_input_documents_link",
        clean.get("input_documents_sha256") == acquired_documents_sha256,
        str(clean.get("input_documents_sha256")),
    )
    _record(
        checks,
        "cleaning_input_manifest_link",
        clean.get("input_manifest_sha256") == acquired_manifest_sha256,
        str(clean.get("input_manifest_sha256")),
    )
    _record(
        checks,
        "deduplication_input_documents_link",
        deduplication.get("input_documents_sha256") == clean_documents_sha256,
        str(deduplication.get("input_documents_sha256")),
    )
    _record(
        checks,
        "deduplication_input_manifest_link",
        deduplication.get("input_manifest_sha256") == clean_manifest_sha256,
        str(deduplication.get("input_manifest_sha256")),
    )
    _record(
        checks,
        "splitting_input_documents_link",
        split.get("input_documents_sha256") == clean_documents_sha256,
        str(split.get("input_documents_sha256")),
    )
    _record(
        checks,
        "splitting_input_manifest_link",
        split.get("input_manifest_sha256") == clean_manifest_sha256,
        str(split.get("input_manifest_sha256")),
    )
    _record(
        checks,
        "splitting_deduplication_manifest_link",
        split.get("deduplication_manifest_sha256") == deduplication_manifest_sha256,
        str(split.get("deduplication_manifest_sha256")),
    )
    _record(
        checks,
        "splitting_duplicate_clusters_link",
        split.get("duplicate_clusters_sha256") == clusters_sha256,
        str(split.get("duplicate_clusters_sha256")),
    )
    _record(
        checks,
        "cleaning_dropped_zero",
        clean.get("dropped_count") == 0,
        str(clean.get("dropped_count")),
    )
    _record(
        checks,
        "deduplication_report_only",
        deduplication.get("action") == "report_only",
        str(deduplication.get("action")),
    )
    _record(
        checks,
        "deduplication_dropped_zero",
        deduplication.get("dropped_count") == 0,
        str(deduplication.get("dropped_count")),
    )
    _record(
        checks,
        "split_frozen",
        split.get("frozen") is True,
        str(split.get("frozen")),
    )
    _record(
        checks,
        "split_no_overlap",
        split.get("overlap_document_count") == 0,
        str(split.get("overlap_document_count")),
    )
    _record(
        checks,
        "split_no_cross_duplicate_clusters",
        split.get("cross_split_duplicate_cluster_count") == 0,
        str(split.get("cross_split_duplicate_cluster_count")),
    )
    _record(
        checks,
        "privacy_policy_warn",
        acquired.get("privacy_action") == "warn"
        and clean.get("transform", {}).get("privacy_action") == "warn",
        f"acquired={acquired.get('privacy_action')}, clean={clean.get('transform', {}).get('privacy_action')}",
    )
    _record(
        checks,
        "quality_policy_warn",
        clean.get("transform", {}).get("quality_action") == "warn",
        str(clean.get("transform", {}).get("quality_action")),
    )
    _record(
        checks,
        "language_bucket_coverage",
        acquired.get("uncovered_language_buckets") == [],
        str(acquired.get("uncovered_language_buckets")),
    )
    _record(
        checks,
        "content_bucket_coverage",
        acquired.get("uncovered_content_buckets") == [],
        str(acquired.get("uncovered_content_buckets")),
    )

    language_tokens = acquired.get("estimated_tokens_by_language_bucket")
    content_tokens = acquired.get("estimated_tokens_by_content")
    source_tokens = acquired.get("estimated_tokens_by_source")
    if (
        not isinstance(language_tokens, dict)
        or not isinstance(content_tokens, dict)
        or not isinstance(source_tokens, dict)
    ):
        raise FormalAuditError("acquisition manifest is missing token distribution")
    total_tokens = acquired.get("estimated_tokens")
    if type(total_tokens) is not int or total_tokens <= 0:
        raise FormalAuditError("acquisition estimated_tokens is invalid")

    language_order_ok = all(
        language_tokens[left] > language_tokens[right]
        for left, right in zip(LANGUAGE_PRIORITY, LANGUAGE_PRIORITY[1:], strict=False)
    )
    _record(
        checks,
        "language_priority_order",
        language_order_ok,
        " > ".join(f"{name}={language_tokens.get(name)}" for name in LANGUAGE_PRIORITY),
    )
    _record(
        checks,
        "content_bucket_set",
        set(content_tokens) == set(mixture.content_mix),
        str(sorted(content_tokens)),
    )
    max_source_fraction = mixture.constraints.max_source_fraction
    source_fractions = _fraction_map(source_tokens, total_tokens)
    source_limit_ok = all(
        fraction <= max_source_fraction for fraction in source_fractions.values()
    )
    _record(
        checks,
        "source_fraction_limit",
        source_limit_ok,
        f"limit={max_source_fraction}, fractions={source_fractions}",
    )

    exact_duplicate_fraction = (
        deduplication.get("exact_duplicate_document_count", 0)
        / deduplication["record_count"]
    )
    near_duplicate_fraction = (
        deduplication.get("near_candidate_document_count", 0)
        / deduplication["record_count"]
    )
    _record(
        checks,
        "exact_duplicate_fraction_limit",
        exact_duplicate_fraction <= mixture.constraints.max_exact_duplicate_fraction,
        f"{exact_duplicate_fraction:.6f}",
    )
    _record(
        checks,
        "near_duplicate_fraction_limit",
        near_duplicate_fraction <= mixture.constraints.max_near_duplicate_fraction,
        f"{near_duplicate_fraction:.6f}",
    )

    failures = [check for check in checks if not check["passed"]]
    status = "passed" if not failures else "blocked"
    manifest = {
        "schema_version": 1,
        "audit_version": FORMAL_AUDIT_VERSION,
        "status": status,
        "formal_training_eligible": status == "passed",
        "plan_id": plan.plan_id,
        "mixture_plan_id": mixture.plan_id,
        "estimated_tokens": total_tokens,
        "record_count": acquired["record_count"],
        "manifest_sha256": {
            "acquisition": acquired_manifest_sha256,
            "cleaning": clean_manifest_sha256,
            "deduplication": deduplication_manifest_sha256,
            "splitting": split_manifest_sha256,
            "processing": processing_manifest_sha256,
        },
        "distributions": {
            "language_token_fractions": _fraction_map(language_tokens, total_tokens),
            "content_token_fractions": _fraction_map(content_tokens, total_tokens),
            "source_token_fractions": source_fractions,
        },
        "warning_counts": {
            "privacy": clean.get("privacy_warning_counts", {}),
            "quality": clean.get("quality_warning_counts", {}),
        },
        "deduplication": {
            "exact_duplicate_fraction": round(exact_duplicate_fraction, 6),
            "near_duplicate_fraction": round(near_duplicate_fraction, 6),
            "exact_cluster_count": deduplication["exact_cluster_count"],
            "near_cluster_count": deduplication["near_cluster_count"],
        },
        "splitting": {
            "split_counts": split["split_counts"],
            "overlap_document_count": split["overlap_document_count"],
            "cross_split_duplicate_cluster_count": split[
                "cross_split_duplicate_cluster_count"
            ],
        },
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(audit_path / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit formal-data v0 lineage, distributions, and release gates."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--mixture-path", type=Path, default=DEFAULT_MIXTURE_PATH)
    parser.add_argument("--acquired-dir", type=Path, default=DEFAULT_ACQUIRED_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument(
        "--deduplication-dir",
        type=Path,
        default=DEFAULT_DEDUPLICATION_DIR,
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--processing-dir", type=Path, default=DEFAULT_PROCESSING_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = audit_formal_v0(
        project_root=args.project_root,
        plan_path=args.plan_path,
        mixture_path=args.mixture_path,
        acquired_dir=args.acquired_dir,
        clean_dir=args.clean_dir,
        deduplication_dir=args.deduplication_dir,
        split_dir=args.split_dir,
        processing_dir=args.processing_dir,
        audit_dir=args.audit_dir,
    )
    print(
        json.dumps(
            {
                "failure_count": manifest["failure_count"],
                "formal_training_eligible": manifest["formal_training_eligible"],
                "status": manifest["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
