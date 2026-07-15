"""Reproducible Hugging Face snapshot downloads for post-training stages."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download


SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_VALID_STAGES = frozenset(range(8, 13))


class PostTrainingDownloadError(RuntimeError):
    """Raised when a download contract or local snapshot is invalid."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise PostTrainingDownloadError(f"{context} must be a mapping")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise PostTrainingDownloadError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise PostTrainingDownloadError(
            f"{context} has unknown fields: {', '.join(unknown)}"
        )


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PostTrainingDownloadError(f"{field} must be a non-empty string")
    return value


def _safe_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise PostTrainingDownloadError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PostTrainingDownloadError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    repo_id: str
    revision: str
    stages: tuple[int, ...]
    local_dir: Path
    license: str | None
    expected_files: int
    expected_bytes: int
    allow_patterns: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> DatasetSnapshot:
        data = _mapping(value, "dataset")
        _exact_keys(
            data,
            {
                "repo_id",
                "revision",
                "stages",
                "local_dir",
                "license",
                "expected_files",
                "expected_bytes",
                "allow_patterns",
            },
            "dataset",
        )
        revision = _string(data["revision"], "revision")
        if _COMMIT_PATTERN.fullmatch(revision) is None:
            raise PostTrainingDownloadError("revision must be a 40-character commit")
        raw_stages = data["stages"]
        if not isinstance(raw_stages, list) or not raw_stages:
            raise PostTrainingDownloadError("stages must be a non-empty list")
        if any(
            type(stage) is not int or stage not in _VALID_STAGES for stage in raw_stages
        ):
            raise PostTrainingDownloadError("stages must contain integers from 8 to 12")
        stages = tuple(raw_stages)
        if len(set(stages)) != len(stages):
            raise PostTrainingDownloadError("stages must not contain duplicates")
        raw_patterns = data["allow_patterns"]
        if not isinstance(raw_patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in raw_patterns
        ):
            raise PostTrainingDownloadError("allow_patterns must be a string list")
        license_value = data["license"]
        if license_value is not None:
            license_value = _string(license_value, "license")
        return cls(
            repo_id=_string(data["repo_id"], "repo_id"),
            revision=revision,
            stages=stages,
            local_dir=_safe_path(data["local_dir"], "local_dir"),
            license=license_value,
            expected_files=_positive_int(data["expected_files"], "expected_files"),
            expected_bytes=_positive_int(data["expected_bytes"], "expected_bytes"),
            allow_patterns=tuple(raw_patterns),
        )


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    snapshot_id: str
    output_root: Path
    manifest_path: Path
    datasets: tuple[DatasetSnapshot, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> DownloadConfig:
        data = _mapping(value, "config")
        _exact_keys(
            data,
            {
                "schema_version",
                "snapshot_id",
                "output_root",
                "manifest_path",
                "datasets",
            },
            "config",
        )
        if data["schema_version"] != SCHEMA_VERSION:
            raise PostTrainingDownloadError(f"schema_version must be {SCHEMA_VERSION}")
        raw_datasets = data["datasets"]
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise PostTrainingDownloadError("datasets must be a non-empty list")
        datasets = tuple(DatasetSnapshot.from_mapping(item) for item in raw_datasets)
        repo_ids = [item.repo_id for item in datasets]
        local_dirs = [item.local_dir for item in datasets]
        if len(set(repo_ids)) != len(repo_ids):
            raise PostTrainingDownloadError("repo_id values must be unique")
        if len(set(local_dirs)) != len(local_dirs):
            raise PostTrainingDownloadError("local_dir values must be unique")
        covered_stages = {stage for item in datasets for stage in item.stages}
        if covered_stages != _VALID_STAGES:
            raise PostTrainingDownloadError("datasets must cover stages 8 through 12")
        return cls(
            snapshot_id=_string(data["snapshot_id"], "snapshot_id"),
            output_root=_safe_path(data["output_root"], "output_root"),
            manifest_path=_safe_path(data["manifest_path"], "manifest_path"),
            datasets=datasets,
        )


def load_download_config(path: Path) -> DownloadConfig:
    if not path.is_file():
        raise FileNotFoundError(f"post-training download config not found: {path}")
    return DownloadConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))


def select_datasets(
    config: DownloadConfig, stages: set[int] | None
) -> tuple[DatasetSnapshot, ...]:
    if not stages:
        return config.datasets
    invalid = stages - _VALID_STAGES
    if invalid:
        raise PostTrainingDownloadError(f"invalid stages: {sorted(invalid)}")
    return tuple(item for item in config.datasets if stages.intersection(item.stages))


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def verify_snapshot(
    snapshot: DatasetSnapshot,
    output_root: Path,
    api: HfApi,
) -> dict[str, Any]:
    info = api.dataset_info(
        snapshot.repo_id,
        revision=snapshot.revision,
        files_metadata=True,
    )
    if info.sha != snapshot.revision:
        raise PostTrainingDownloadError(
            f"{snapshot.repo_id}: resolved revision {info.sha} does not match contract"
        )
    remote = {
        sibling.rfilename: sibling.size
        for sibling in info.siblings
        if _matches(sibling.rfilename, snapshot.allow_patterns)
    }
    if any(size is None for size in remote.values()):
        raise PostTrainingDownloadError(
            f"{snapshot.repo_id}: upstream did not return all file sizes"
        )
    remote_bytes = sum(remote.values())
    if (
        len(remote) != snapshot.expected_files
        or remote_bytes != snapshot.expected_bytes
    ):
        raise PostTrainingDownloadError(
            f"{snapshot.repo_id}: pinned upstream metadata differs from contract"
        )

    local_dir = output_root / snapshot.local_dir
    local = {
        str(path.relative_to(local_dir)): path.stat().st_size
        for path in local_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(local_dir).parts
    }
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    mismatched = sorted(
        path for path in set(remote) & set(local) if remote[path] != local[path]
    )
    if missing or extra or mismatched:
        raise PostTrainingDownloadError(
            f"{snapshot.repo_id}: verification failed; missing={missing}, "
            f"extra={extra}, mismatched={mismatched}"
        )
    return {
        "repo_id": snapshot.repo_id,
        "revision": snapshot.revision,
        "stages": list(snapshot.stages),
        "local_dir": str(snapshot.local_dir),
        "license": snapshot.license,
        "files": len(local),
        "bytes": sum(local.values()),
        "status": "verified",
    }


def _write_manifest(
    path: Path, config: DownloadConfig, records: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": config.snapshot_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def execute(
    config: DownloadConfig,
    *,
    stages: set[int] | None = None,
    verify_only: bool = False,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    endpoint = os.getenv("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint, token=token)
    selected = select_datasets(config, stages)
    records: list[dict[str, Any]] = []
    for snapshot in selected:
        local_dir = config.output_root / snapshot.local_dir
        if not verify_only:
            snapshot_download(
                repo_id=snapshot.repo_id,
                repo_type="dataset",
                revision=snapshot.revision,
                local_dir=local_dir,
                allow_patterns=list(snapshot.allow_patterns) or None,
                max_workers=max_workers,
                token=token,
                endpoint=endpoint,
            )
        record = verify_snapshot(snapshot, config.output_root, api)
        records.append(record)
        print(
            f"verified {snapshot.repo_id} "
            f"({record['files']} files, {record['bytes']} bytes)"
        )
    _write_manifest(config.manifest_path, config, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/post-training-sources-v1.yaml"),
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=range(8, 13),
        action="append",
        help="download only datasets used by this stage; may be repeated",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files without downloading",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers <= 0:
        raise PostTrainingDownloadError("max-workers must be positive")
    config = load_download_config(args.config)
    execute(
        config,
        stages=set(args.stage) if args.stage else None,
        verify_only=args.verify_only,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
