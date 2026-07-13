"""Build and verify deterministic document-internal long-context views."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from atomllm.training.config import file_sha256
from atomllm.training.formal_token_shards import verify_formal_token_shards


SCHEMA_VERSION = 1
FORMAT_VERSION = "document-long-window-view-v1"
SELECTION_VERSION = "atomllm-long-window-sha256-v1"


class LongWindowViewError(RuntimeError):
    """Raised when a long-window view violates its lineage or budget."""


@dataclass(frozen=True, slots=True)
class LongWindowViewConfig:
    name: str
    stage: str
    source_dir: Path
    output_dir: Path
    window_length: int
    stride: int
    expected_candidate_count: int
    selection_count: int


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LongWindowViewError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise LongWindowViewError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise LongWindowViewError(f"{field} must be a positive integer")
    return value


def load_long_window_view_config(path: str | Path) -> LongWindowViewConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise LongWindowViewError(f"cannot read long-window config: {path}") from error
    required = {
        "schema_version",
        "name",
        "stage",
        "source_dir",
        "output_dir",
        "window_length",
        "stride",
        "expected_candidate_count",
        "selection_count",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise LongWindowViewError("long-window config fields are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise LongWindowViewError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise LongWindowViewError("name must be non-empty")
    if raw["stage"] not in {"B", "C"}:
        raise LongWindowViewError("stage must be B or C")
    config = LongWindowViewConfig(
        name=raw["name"],
        stage=raw["stage"],
        source_dir=_safe_relative_path(raw["source_dir"], "source_dir"),
        output_dir=_safe_relative_path(raw["output_dir"], "output_dir"),
        window_length=_positive_int(raw["window_length"], "window_length"),
        stride=_positive_int(raw["stride"], "stride"),
        expected_candidate_count=_positive_int(
            raw["expected_candidate_count"], "expected_candidate_count"
        ),
        selection_count=_positive_int(raw["selection_count"], "selection_count"),
    )
    if config.stride > config.window_length:
        raise LongWindowViewError("stride must not exceed window_length")
    if config.selection_count > config.expected_candidate_count:
        raise LongWindowViewError("selection_count exceeds candidate count")
    return config


def _selection_sha256(
    source_manifest_sha256: str,
    stage: str,
    shard_index: int,
    document_index: int,
    window_start: int,
    window_end: int,
) -> str:
    identity = "\0".join(
        (
            SELECTION_VERSION,
            source_manifest_sha256,
            stage,
            str(shard_index),
            str(document_index),
            str(window_start),
            str(window_end),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_long_window_view(
    config: LongWindowViewConfig,
    *,
    project_root: str | Path = ".",
) -> Path:
    root = Path(project_root).resolve()
    source_dir = root / config.source_dir
    output_dir = root / config.output_dir
    source_manifest = verify_formal_token_shards(source_dir)
    source_manifest_path = source_dir / "manifest.json"
    source_manifest_sha256 = file_sha256(source_manifest_path)
    candidates: list[dict[str, Any]] = []
    overlap_ratio = (config.window_length - config.stride) / config.window_length
    for shard_index, shard in enumerate(source_manifest["shards"]):
        index_path = source_dir / shard["index_file"]["name"]
        documents = np.memmap(index_path, mode="r", dtype="<u8").reshape(-1, 2)
        for document_index, (document_offset, document_length) in enumerate(documents):
            length = int(document_length)
            if length < config.window_length:
                continue
            for window_start in range(
                0,
                length - config.window_length + 1,
                config.stride,
            ):
                window_end = window_start + config.window_length
                candidates.append(
                    {
                        "source_shard_index": shard_index,
                        "source_token_file": shard["token_file"]["name"],
                        "document_index": document_index,
                        "document_token_offset": int(document_offset),
                        "document_token_count": length,
                        "window_start": window_start,
                        "window_end": window_end,
                        "source_token_offset": int(document_offset) + window_start,
                        "selection_sha256": _selection_sha256(
                            source_manifest_sha256,
                            config.stage,
                            shard_index,
                            document_index,
                            window_start,
                            window_end,
                        ),
                        "overlap_ratio": (overlap_ratio if window_start > 0 else 0.0),
                    }
                )
    if len(candidates) != config.expected_candidate_count:
        raise LongWindowViewError(
            "candidate count mismatch: "
            f"expected {config.expected_candidate_count}, got {len(candidates)}"
        )
    candidates.sort(
        key=lambda item: (
            item["selection_sha256"],
            item["source_shard_index"],
            item["document_index"],
            item["window_start"],
        )
    )
    selected = candidates[: config.selection_count]
    for selection_rank, item in enumerate(selected):
        item["selection_rank"] = selection_rank
    identity = {
        "format_version": FORMAT_VERSION,
        "name": config.name,
        "stage": config.stage,
        "source_dataset_id": source_manifest["dataset_id"],
        "source_manifest_sha256": source_manifest_sha256,
        "selection_version": SELECTION_VERSION,
        "window_length": config.window_length,
        "stride": config.stride,
        "selection_count": config.selection_count,
    }
    dataset_id = f"long-window-{hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:16]}"
    source_relative = Path("..") / source_dir.name
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "dataset_id": dataset_id,
        "dataset_manifest_identity": identity,
        "source_directory": source_relative.as_posix(),
        "source_dataset_id": source_manifest["dataset_id"],
        "source_manifest_sha256": source_manifest_sha256,
        "formal_training_eligible": source_manifest["formal_training_eligible"],
        "stage": config.stage,
        "selection_version": SELECTION_VERSION,
        "window_length": config.window_length,
        "stride": config.stride,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_tokens": len(candidates) * config.window_length,
        "selected_tokens": len(selected) * config.window_length,
        "windows": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        f"{_canonical_json(manifest, pretty=True)}\n",
        encoding="utf-8",
    )
    (output_dir / "COMPLETED").write_text(
        "atomllm-long-window-view-complete-v1\n",
        encoding="utf-8",
    )
    verify_long_window_view(output_dir)
    return manifest_path


def verify_long_window_view(directory: str | Path) -> dict[str, Any]:
    view_dir = Path(directory)
    manifest_path = view_dir / "manifest.json"
    completed = view_dir / "COMPLETED"
    if not manifest_path.is_file() or not completed.is_file():
        raise LongWindowViewError("long-window view is incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LongWindowViewError("long-window manifest is invalid") from error
    if manifest.get("format_version") != FORMAT_VERSION:
        raise LongWindowViewError("long-window format version is invalid")
    window_length = manifest.get("window_length")
    windows = manifest.get("windows")
    if type(window_length) is not int or window_length <= 0:
        raise LongWindowViewError("long-window length is invalid")
    if not isinstance(windows, list) or len(windows) != manifest.get("selected_count"):
        raise LongWindowViewError("selected window count is invalid")
    source_dir = (view_dir / manifest.get("source_directory", "")).resolve()
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise LongWindowViewError("source token-shard manifest is missing")
    if file_sha256(source_manifest_path) != manifest.get("source_manifest_sha256"):
        raise LongWindowViewError("source token-shard manifest SHA-256 mismatch")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("dataset_id") != manifest.get("source_dataset_id"):
        raise LongWindowViewError("source dataset ID mismatch")
    previous_hash = ""
    seen: set[tuple[int, int, int]] = set()
    source_indices = [
        np.memmap(
            source_dir / shard["index_file"]["name"],
            mode="r",
            dtype="<u8",
        ).reshape(-1, 2)
        for shard in source["shards"]
    ]
    for rank, item in enumerate(windows):
        if not isinstance(item, dict) or item.get("selection_rank") != rank:
            raise LongWindowViewError("window selection rank is invalid")
        selection_hash = item.get("selection_sha256")
        if not isinstance(selection_hash, str) or len(selection_hash) != 64:
            raise LongWindowViewError("window selection hash is invalid")
        if selection_hash < previous_hash:
            raise LongWindowViewError("windows are not in deterministic hash order")
        previous_hash = selection_hash
        key = (
            item.get("source_shard_index"),
            item.get("document_index"),
            item.get("window_start"),
        )
        if key in seen:
            raise LongWindowViewError("long-window view contains duplicates")
        seen.add(key)
        shard_index = item.get("source_shard_index")
        if type(shard_index) is not int or not 0 <= shard_index < len(source["shards"]):
            raise LongWindowViewError("window source shard is invalid")
        document_index = item.get("document_index")
        if type(document_index) is not int or not 0 <= document_index < len(
            source_indices[shard_index]
        ):
            raise LongWindowViewError("window source document is invalid")
        actual_offset, actual_length = source_indices[shard_index][document_index]
        if (
            item.get("source_token_file")
            != source["shards"][shard_index]["token_file"]["name"]
        ):
            raise LongWindowViewError("window source token file is invalid")
        if item.get("document_token_offset") != int(actual_offset):
            raise LongWindowViewError("window document offset is invalid")
        if item.get("document_token_count") != int(actual_length):
            raise LongWindowViewError("window document length is invalid")
        if item.get("window_end") - item.get("window_start") != window_length:
            raise LongWindowViewError("window length is inconsistent")
        if item.get("window_end") > item.get("document_token_count"):
            raise LongWindowViewError("window crosses a document boundary")
        if item.get("source_token_offset") != int(actual_offset) + item.get(
            "window_start"
        ):
            raise LongWindowViewError("window source token offset is invalid")
        expected_hash = _selection_sha256(
            manifest["source_manifest_sha256"],
            manifest["stage"],
            shard_index,
            item["document_index"],
            item["window_start"],
            item["window_end"],
        )
        if selection_hash != expected_hash:
            raise LongWindowViewError("window selection hash mismatch")
    if manifest.get("selected_tokens") != len(windows) * window_length:
        raise LongWindowViewError("selected token count is inconsistent")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic document-internal long-window view."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--verify", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify is not None:
        manifest = verify_long_window_view(args.verify)
        print(_canonical_json(manifest["dataset_manifest_identity"], pretty=True))
        return 0
    config = load_long_window_view_config(args.config)
    path = build_long_window_view(config, project_root=args.project_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
