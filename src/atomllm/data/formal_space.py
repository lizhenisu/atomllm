"""Low-memory formal-data space-version acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from atomllm.data.formal_acquisition import (
    TOKEN_ESTIMATOR,
    FormalAcquisitionError,
    StreamSpec,
    _document_from_record,
    _iter_huggingface,
    language_bucket,
)
from atomllm.data.formal_plan import load_formal_data_plan
from atomllm.data.mixture import load_pretraining_mixture


FORMAL_SPACE_SCHEMA_VERSION = 1
FORMAL_SPACE_VERSION = "formal-70g-space-v1"
DEFAULT_CONFIG_PATH = Path("configs/data/formal-70g-space-acquisition.yaml")
GIB = 1024**3
LANGUAGE_ORDER = ("zh-Hans", "en", "zh-Hant", "ja", "other")


class FormalSpaceError(RuntimeError):
    """Raised when the formal space acquisition cannot safely continue."""


def _format_gib(value: int) -> str:
    return f"{value / GIB:.3f}GiB"


def _format_rate(value: float) -> str:
    if value >= GIB:
        return f"{value / GIB:.3f}GiB/s"
    if value >= 1024**2:
        return f"{value / (1024**2):.2f}MiB/s"
    if value >= 1024:
        return f"{value / 1024:.2f}KiB/s"
    return f"{value:.0f}B/s"


def _emit_progress(
    message: str,
    *,
    progress: bool,
) -> None:
    if progress:
        print(message, file=sys.stderr, flush=True)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormalSpaceError(f"{context} must be a mapping")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise FormalSpaceError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise FormalSpaceError(f"{context} has unknown fields: {', '.join(unknown)}")


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalSpaceError(f"{field_name} must be a non-empty string")
    return value


def _safe_relative_path(value: Any, field_name: str) -> Path:
    path = Path(_string(value, field_name))
    if path.is_absolute() or ".." in path.parts:
        raise FormalSpaceError(f"{field_name} must be a safe relative path")
    return path


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise FormalSpaceError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise FormalSpaceError(f"{field_name} must be a non-negative integer")
    return value


def _sha256_and_line_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
    return digest.hexdigest(), line_count


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalSpaceError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalSpaceError(f"JSON file must contain an object: {path}")
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


def _counter_from_state(value: Any, field_name: str) -> Counter[str]:
    mapping = _mapping(value, field_name)
    counter: Counter[str] = Counter()
    for key, item in mapping.items():
        if type(item) is not int or item < 0:
            raise FormalSpaceError(f"{field_name}.{key} must be a non-negative int")
        counter[key] = item
    return counter


@dataclass(frozen=True, slots=True)
class SpaceStreamSpec:
    stream: StreamSpec
    target_document_bytes: int

    @classmethod
    def from_mapping(cls, value: Any) -> SpaceStreamSpec:
        data = _mapping(value, "space stream")
        target_document_bytes = _positive_int(
            data.get("target_document_bytes"),
            "target_document_bytes",
        )
        stream_mapping = dict(data)
        del stream_mapping["target_document_bytes"]
        stream_mapping["target_estimated_tokens"] = 1
        try:
            stream = StreamSpec.from_mapping(stream_mapping)
        except FormalAcquisitionError as error:
            raise FormalSpaceError(str(error)) from error
        return cls(stream=stream, target_document_bytes=target_document_bytes)

    def to_mapping(self) -> dict[str, Any]:
        mapping = self.stream.to_mapping()
        del mapping["target_estimated_tokens"]
        mapping["target_document_bytes"] = self.target_document_bytes
        return mapping


@dataclass(frozen=True, slots=True)
class FormalSpaceConfig:
    schema_version: int
    plan_id: str
    formal_plan_path: Path
    target_document_bytes: int
    output_dir: Path
    minimum_free_bytes: int
    checkpoint_every_records: int
    checkpoint_every_bytes: int
    source_target_ceiling_bytes: int
    exhaustion_fallback_stream_ids: tuple[str, ...]
    streams: tuple[SpaceStreamSpec, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> FormalSpaceConfig:
        data = _mapping(value, "formal space config")
        _exact_keys(
            data,
            {
                "schema_version",
                "plan_id",
                "formal_plan_path",
                "target_document_bytes",
                "output_dir",
                "minimum_free_bytes",
                "checkpoint_every_records",
                "checkpoint_every_bytes",
                "source_target_ceiling_bytes",
                "exhaustion_fallback_stream_ids",
                "streams",
            },
            "formal space config",
        )
        if data["schema_version"] != FORMAL_SPACE_SCHEMA_VERSION:
            raise FormalSpaceError("schema_version must be 1")
        raw_streams = data["streams"]
        if not isinstance(raw_streams, list) or not raw_streams:
            raise FormalSpaceError("streams must be a non-empty list")
        streams = tuple(SpaceStreamSpec.from_mapping(item) for item in raw_streams)
        stream_ids = [item.stream.stream_id for item in streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise FormalSpaceError("stream_id values must be unique")
        fallback_stream_ids = data["exhaustion_fallback_stream_ids"]
        if not isinstance(fallback_stream_ids, list) or not fallback_stream_ids:
            raise FormalSpaceError(
                "exhaustion_fallback_stream_ids must be a non-empty list"
            )
        if not all(isinstance(item, str) and item for item in fallback_stream_ids):
            raise FormalSpaceError(
                "exhaustion_fallback_stream_ids must contain strings"
            )
        if len(fallback_stream_ids) != len(set(fallback_stream_ids)):
            raise FormalSpaceError("exhaustion_fallback_stream_ids must be unique")
        unknown_fallbacks = sorted(set(fallback_stream_ids) - set(stream_ids))
        if unknown_fallbacks:
            raise FormalSpaceError(
                "exhaustion_fallback_stream_ids reference unknown streams: "
                f"{', '.join(unknown_fallbacks)}"
            )
        target = _positive_int(data["target_document_bytes"], "target_document_bytes")
        if sum(stream.target_document_bytes for stream in streams) != target:
            raise FormalSpaceError("stream targets must sum to target_document_bytes")
        source_target_ceiling_bytes = _positive_int(
            data["source_target_ceiling_bytes"],
            "source_target_ceiling_bytes",
        )
        if source_target_ceiling_bytes > target:
            raise FormalSpaceError(
                "source_target_ceiling_bytes must not exceed target_document_bytes"
            )
        return cls(
            schema_version=FORMAL_SPACE_SCHEMA_VERSION,
            plan_id=_string(data["plan_id"], "plan_id"),
            formal_plan_path=_safe_relative_path(
                data["formal_plan_path"],
                "formal_plan_path",
            ),
            target_document_bytes=target,
            output_dir=_safe_relative_path(data["output_dir"], "output_dir"),
            minimum_free_bytes=_non_negative_int(
                data["minimum_free_bytes"],
                "minimum_free_bytes",
            ),
            checkpoint_every_records=_positive_int(
                data["checkpoint_every_records"],
                "checkpoint_every_records",
            ),
            checkpoint_every_bytes=_positive_int(
                data["checkpoint_every_bytes"],
                "checkpoint_every_bytes",
            ),
            source_target_ceiling_bytes=source_target_ceiling_bytes,
            exhaustion_fallback_stream_ids=tuple(fallback_stream_ids),
            streams=streams,
        )


def load_formal_space_config(path: str | Path) -> FormalSpaceConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"formal space config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise FormalSpaceError(f"invalid formal space YAML: {error}") from error
    return FormalSpaceConfig.from_mapping(raw)


def _source_byte_targets(config: FormalSpaceConfig) -> dict[str, int]:
    targets: Counter[str] = Counter()
    for item in config.streams:
        targets[item.stream.source_id] += item.target_document_bytes
    return dict(sorted(targets.items()))


def _validate_plan_and_mixture(
    config: FormalSpaceConfig,
    project_root: Path,
) -> dict[str, Any]:
    formal_plan = load_formal_data_plan(
        project_root / config.formal_plan_path,
        project_root=project_root,
    )
    if not formal_plan.training_eligible:
        raise FormalSpaceError("formal v0 plan is not training_eligible")
    mixture = load_pretraining_mixture(
        project_root / "configs/data/pretraining-mixture.yaml"
    )
    source_targets = _source_byte_targets(config)
    if config.source_target_ceiling_bytes > int(
        config.target_document_bytes * mixture.constraints.max_source_fraction
    ):
        raise FormalSpaceError(
            "source_target_ceiling_bytes exceeds the pretraining mixture source limit"
        )
    ceiling_violations = {
        source: bytes_
        for source, bytes_ in source_targets.items()
        if bytes_ > config.source_target_ceiling_bytes
    }
    if ceiling_violations:
        raise FormalSpaceError(
            "source byte target exceeds source_target_ceiling_bytes: "
            f"{ceiling_violations}"
        )
    source_fractions = {
        source: bytes_ / config.target_document_bytes
        for source, bytes_ in source_targets.items()
    }
    too_large = {
        source: fraction
        for source, fraction in source_fractions.items()
        if fraction > mixture.constraints.max_source_fraction
    }
    if too_large:
        raise FormalSpaceError(f"source byte target exceeds limit: {too_large}")
    content_buckets = {item.stream.content_type for item in config.streams}
    if content_buckets != set(mixture.content_mix):
        raise FormalSpaceError("streams do not cover all pretraining content buckets")
    language_buckets = {
        language_bucket(item.stream.language) for item in config.streams
    }
    if "auto-zh-script" in {item.stream.language for item in config.streams}:
        language_buckets.update({"zh-Hans", "zh-Hant"})
    if set(mixture.language_mix) - language_buckets:
        raise FormalSpaceError("streams do not cover all pretraining language buckets")
    return {
        "formal_data_plan_id": formal_plan.plan_id,
        "mixture_plan_id": mixture.plan_id,
        "max_source_fraction": mixture.constraints.max_source_fraction,
        "source_target_ceiling_bytes": config.source_target_ceiling_bytes,
        "exhaustion_fallback_stream_ids": list(config.exhaustion_fallback_stream_ids),
        "source_byte_targets": source_targets,
        "source_byte_fractions": {
            source: round(fraction, 6)
            for source, fraction in sorted(source_fractions.items())
        },
    }


def plan_formal_space(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate and summarize the space acquisition plan without downloading."""
    root = Path(project_root)
    config = load_formal_space_config(root / config_path)
    validation = _validate_plan_and_mixture(config, root)
    output_dir = root / config.output_dir
    usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else root)
    return {
        "schema_version": FORMAL_SPACE_SCHEMA_VERSION,
        "space_version": FORMAL_SPACE_VERSION,
        "plan_id": config.plan_id,
        "target_document_bytes": config.target_document_bytes,
        "target_document_gib": round(config.target_document_bytes / GIB, 3),
        "output_dir": config.output_dir.as_posix(),
        "minimum_free_bytes": config.minimum_free_bytes,
        "available_free_bytes": usage.free,
        "disk_space_ready": usage.free >= config.minimum_free_bytes,
        "checkpoint_every_records": config.checkpoint_every_records,
        "checkpoint_every_bytes": config.checkpoint_every_bytes,
        "source_target_ceiling_bytes": config.source_target_ceiling_bytes,
        "exhaustion_fallback_stream_ids": list(config.exhaustion_fallback_stream_ids),
        "stream_count": len(config.streams),
        "streams": [stream.to_mapping() for stream in config.streams],
        **validation,
    }


def _initial_state(
    config: FormalSpaceConfig,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_SPACE_SCHEMA_VERSION,
        "space_version": FORMAL_SPACE_VERSION,
        "plan_id": config.plan_id,
        "formal_data_plan_id": validation["formal_data_plan_id"],
        "target_document_bytes": config.target_document_bytes,
        "stream_positions": {
            item.stream.stream_id: item.stream.initial_skip_records
            for item in config.streams
        },
        "stream_document_bytes": {item.stream.stream_id: 0 for item in config.streams},
        "stream_effective_target_bytes": {
            item.stream.stream_id: item.target_document_bytes for item in config.streams
        },
        "exhausted_streams": {},
        "redistributions": [],
        "records_written": 0,
        "documents_size_bytes": 0,
        "estimated_tokens": 0,
        "language_counts": {},
        "content_counts": {},
        "source_counts": {},
        "privacy_warning_counts": {},
        "estimated_tokens_by_language_bucket": {},
        "estimated_tokens_by_content": {},
        "estimated_tokens_by_source": {},
        "completed": False,
    }


def _load_or_create_state(
    output_dir: Path,
    config: FormalSpaceConfig,
    validation: dict[str, Any],
    *,
    rebase_effective_targets: bool = False,
) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    documents_path = output_dir / "documents.jsonl"
    expected = _initial_state(config, validation)
    if not state_path.exists():
        if documents_path.exists():
            raise FormalSpaceError("documents.jsonl exists without state.json")
        _write_json_atomic(state_path, expected)
        return expected
    state = _read_json(state_path)
    for key in (
        "schema_version",
        "space_version",
        "plan_id",
        "formal_data_plan_id",
        "target_document_bytes",
    ):
        if state.get(key) != expected[key]:
            raise FormalSpaceError(f"existing state mismatch: {key}")
    committed_size = state.get("documents_size_bytes")
    if type(committed_size) is not int or committed_size < 0:
        raise FormalSpaceError("state documents_size_bytes is invalid")
    if documents_path.exists() and documents_path.stat().st_size > committed_size:
        with documents_path.open("ab") as handle:
            handle.truncate(committed_size)
    _migrate_state_for_config(
        state,
        config,
        rebase_effective_targets=rebase_effective_targets,
    )
    _write_json_atomic(state_path, state)
    return state


def _migrate_state_for_config(
    state: dict[str, Any],
    config: FormalSpaceConfig,
    *,
    rebase_effective_targets: bool = False,
) -> None:
    """Add resumable allocation fields to states created by earlier code."""
    stream_ids = {item.stream.stream_id for item in config.streams}
    for field_name in ("stream_positions", "stream_document_bytes"):
        current = _mapping(state.get(field_name), field_name)
        unknown_stream_ids = set(current) - stream_ids
        if unknown_stream_ids:
            raise FormalSpaceError(f"existing state {field_name} does not match config")
        for item in config.streams:
            default = (
                item.stream.initial_skip_records
                if field_name == "stream_positions"
                else 0
            )
            current.setdefault(item.stream.stream_id, default)
    configured_targets = {
        item.stream.stream_id: item.target_document_bytes for item in config.streams
    }
    effective_targets = state.get("stream_effective_target_bytes")
    if effective_targets is None:
        state["stream_effective_target_bytes"] = configured_targets
    else:
        effective_targets = _mapping(
            effective_targets,
            "stream_effective_target_bytes",
        )
        if set(effective_targets) - stream_ids or any(
            type(value) is not int or value < 0 for value in effective_targets.values()
        ):
            raise FormalSpaceError(
                "existing state stream_effective_target_bytes does not match config"
            )
        for item in config.streams:
            effective_targets.setdefault(
                item.stream.stream_id,
                item.target_document_bytes,
            )
    if rebase_effective_targets:
        stream_bytes = _mapping(
            state["stream_document_bytes"],
            "stream_document_bytes",
        )
        smaller_than_written = {
            item.stream.stream_id: stream_bytes[item.stream.stream_id]
            for item in config.streams
            if item.target_document_bytes < stream_bytes[item.stream.stream_id]
        }
        if smaller_than_written:
            raise FormalSpaceError(
                "cannot rebase an effective target below committed bytes: "
                f"{smaller_than_written}"
            )
        state["stream_effective_target_bytes"] = configured_targets
    exhausted_streams = state.setdefault("exhausted_streams", {})
    if not isinstance(exhausted_streams, dict):
        raise FormalSpaceError("existing state exhausted_streams is invalid")
    redistributions = state.setdefault("redistributions", [])
    if not isinstance(redistributions, list):
        raise FormalSpaceError("existing state redistributions is invalid")


def _effective_target(
    state: dict[str, Any],
    stream_id: str,
) -> int:
    targets = _mapping(
        state["stream_effective_target_bytes"],
        "stream_effective_target_bytes",
    )
    target = targets.get(stream_id)
    if type(target) is not int or target < 0:
        raise FormalSpaceError(f"effective target is invalid for stream: {stream_id}")
    return target


def _effective_source_targets(
    state: dict[str, Any],
    config: FormalSpaceConfig,
) -> Counter[str]:
    targets: Counter[str] = Counter()
    for item in config.streams:
        targets[item.stream.source_id] += _effective_target(
            state, item.stream.stream_id
        )
    return targets


def _redistribute_exhausted_stream(
    state: dict[str, Any],
    config: FormalSpaceConfig,
    *,
    exhausted_item: SpaceStreamSpec,
    actual_bytes: int,
    stream_position: int,
) -> list[dict[str, Any]]:
    """Commit an exhausted stream and move its unfinished bytes to safe fallbacks."""
    stream_id = exhausted_item.stream.stream_id
    previous_target = _effective_target(state, stream_id)
    if actual_bytes >= previous_target:
        return []
    effective_targets = _mapping(
        state["stream_effective_target_bytes"],
        "stream_effective_target_bytes",
    )
    effective_targets[stream_id] = actual_bytes
    missing_bytes = previous_target - actual_bytes
    source_targets = _effective_source_targets(state, config)
    streams_by_id = {item.stream.stream_id: item for item in config.streams}
    stream_positions = {
        item.stream.stream_id: index for index, item in enumerate(config.streams)
    }
    exhausted_position = stream_positions[stream_id]
    allocations: list[dict[str, Any]] = []
    for fallback_id in config.exhaustion_fallback_stream_ids:
        if missing_bytes == 0 or stream_positions[fallback_id] <= exhausted_position:
            continue
        fallback = streams_by_id[fallback_id]
        source_headroom = (
            config.source_target_ceiling_bytes
            - source_targets[fallback.stream.source_id]
        )
        if source_headroom <= 0:
            continue
        assigned = min(missing_bytes, source_headroom)
        effective_targets[fallback_id] += assigned
        source_targets[fallback.stream.source_id] += assigned
        missing_bytes -= assigned
        allocations.append(
            {
                "from_stream_id": stream_id,
                "to_stream_id": fallback_id,
                "bytes": assigned,
                "reason": "stream_exhausted",
            }
        )
    if missing_bytes:
        effective_targets[stream_id] = previous_target
        for allocation in allocations:
            effective_targets[allocation["to_stream_id"]] -= allocation["bytes"]
        raise FormalSpaceError(
            "stream exhausted but no safe fallback capacity remains: "
            f"stream_id={stream_id}, source_id={exhausted_item.stream.source_id}, "
            f"written_bytes={actual_bytes}, target_bytes={previous_target}, "
            f"unallocated_bytes={missing_bytes}"
        )
    exhausted_streams = _mapping(state["exhausted_streams"], "exhausted_streams")
    exhausted_streams[stream_id] = {
        "actual_document_bytes": actual_bytes,
        "effective_target_before_exhaustion": previous_target,
        "stream_position": stream_position,
    }
    state["exhausted_streams"] = dict(sorted(exhausted_streams.items()))
    redistributions = state["redistributions"]
    if not isinstance(redistributions, list):
        raise FormalSpaceError("state redistributions is invalid")
    redistributions.extend(allocations)
    return allocations


def _validate_completed_distribution(
    state: dict[str, Any],
    config: FormalSpaceConfig,
    validation: dict[str, Any],
) -> dict[str, Any]:
    source_bytes: Counter[str] = Counter()
    stream_bytes = _mapping(state["stream_document_bytes"], "stream_document_bytes")
    for item in config.streams:
        written = stream_bytes.get(item.stream.stream_id)
        if type(written) is not int or written < 0:
            raise FormalSpaceError("state stream_document_bytes is invalid")
        if written < _effective_target(state, item.stream.stream_id):
            raise FormalSpaceError(
                "completed data has an unfinished effective stream target: "
                f"stream_id={item.stream.stream_id}"
            )
        source_bytes[item.stream.source_id] += written
    max_source_fraction = validation["max_source_fraction"]
    too_large = {
        source: bytes_ / config.target_document_bytes
        for source, bytes_ in source_bytes.items()
        if bytes_ / config.target_document_bytes > max_source_fraction
    }
    if too_large:
        raise FormalSpaceError(f"actual source fraction exceeds limit: {too_large}")
    language_tokens = _counter_from_state(
        state["estimated_tokens_by_language_bucket"],
        "estimated_tokens_by_language_bucket",
    )
    missing_buckets = [
        bucket for bucket in LANGUAGE_ORDER if language_tokens[bucket] <= 0
    ]
    if missing_buckets:
        raise FormalSpaceError(
            "completed data is missing language buckets: " + ", ".join(missing_buckets)
        )
    if any(
        language_tokens[left] <= language_tokens[right]
        for left, right in zip(LANGUAGE_ORDER, LANGUAGE_ORDER[1:])
    ):
        raise FormalSpaceError(
            "completed data violates language priority: "
            + " > ".join(
                f"{bucket}={language_tokens[bucket]}" for bucket in LANGUAGE_ORDER
            )
        )
    return {
        "actual_source_document_bytes": dict(sorted(source_bytes.items())),
        "actual_source_document_fractions": {
            source: round(bytes_ / config.target_document_bytes, 6)
            for source, bytes_ in sorted(source_bytes.items())
        },
        "language_priority": list(LANGUAGE_ORDER),
        "language_priority_passed": True,
    }


def _checkpoint_state(
    state_path: Path,
    handle,
    state: dict[str, Any],
) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    state["documents_size_bytes"] = handle.tell()
    _write_json_atomic(state_path, state)


def _update_counters(
    state: dict[str, Any],
    stream: StreamSpec,
    document,
    line_bytes: int,
    estimated_tokens: int,
) -> None:
    state["records_written"] += 1
    state["estimated_tokens"] += estimated_tokens
    for field_name, key in (
        ("language_counts", document.language),
        ("content_counts", document.content_type),
        ("source_counts", document.source_id),
    ):
        counter = _counter_from_state(state[field_name], field_name)
        counter[key] += 1
        state[field_name] = dict(sorted(counter.items()))
    privacy_counter = _counter_from_state(
        state["privacy_warning_counts"],
        "privacy_warning_counts",
    )
    privacy_counter.update(document.privacy_warnings)
    state["privacy_warning_counts"] = dict(sorted(privacy_counter.items()))
    for field_name, key in (
        ("estimated_tokens_by_language_bucket", language_bucket(document.language)),
        ("estimated_tokens_by_content", document.content_type),
        ("estimated_tokens_by_source", stream.source_id),
    ):
        counter = _counter_from_state(state[field_name], field_name)
        counter[key] += estimated_tokens
        state[field_name] = dict(sorted(counter.items()))
    stream_bytes = _counter_from_state(
        state["stream_document_bytes"],
        "stream_document_bytes",
    )
    stream_bytes[stream.stream_id] += line_bytes
    state["stream_document_bytes"] = dict(sorted(stream_bytes.items()))


def acquire_formal_space(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path = ".",
    require_disk_space: bool = True,
    progress: bool = True,
    rebase_effective_targets: bool = False,
) -> dict[str, Any]:
    """Acquire the formal space version using bounded-memory streaming writes."""
    root = Path(project_root)
    config = load_formal_space_config(root / config_path)
    validation = _validate_plan_and_mixture(config, root)
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if require_disk_space:
        usage = shutil.disk_usage(output_dir)
        if usage.free < config.minimum_free_bytes:
            raise FormalSpaceError(
                "not enough free disk space: "
                f"required={config.minimum_free_bytes}, available={usage.free}"
            )
    documents_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"
    state_path = output_dir / "state.json"
    state = _load_or_create_state(
        output_dir,
        config,
        validation,
        rebase_effective_targets=rebase_effective_targets,
    )
    if state.get("completed") is True:
        if not manifest_path.is_file():
            raise FormalSpaceError("completed state is missing manifest.json")
        _emit_progress(
            f"[formal-space] already completed: {manifest_path.as_posix()}",
            progress=progress,
        )
        return _read_json(manifest_path)

    mode = "ab" if state["documents_size_bytes"] else "wb"
    last_checkpoint_record = state["records_written"]
    last_checkpoint_byte = state["documents_size_bytes"]
    started_at = time.monotonic()

    def report_progress(
        stream: StreamSpec | None, stream_bytes: int, event: str
    ) -> None:
        written = int(state["documents_size_bytes"])
        elapsed = max(time.monotonic() - started_at, 0.001)
        percent = written / config.target_document_bytes * 100
        rate = _format_rate(written / elapsed)
        stream_part = ""
        if stream is not None:
            stream_target = _effective_target(state, stream.stream_id)
            stream_percent = stream_bytes / stream_target * 100
            stream_part = (
                f" stream={stream.stream_id}"
                f" stream_progress={_format_gib(stream_bytes)}/"
                f"{_format_gib(stream_target)}({stream_percent:.2f}%)"
            )
        _emit_progress(
            "[formal-space] "
            f"{event} total={_format_gib(written)}/"
            f"{_format_gib(config.target_document_bytes)}({percent:.2f}%)"
            f" records={state['records_written']}"
            f" tokens≈{state['estimated_tokens']}"
            f" rate={rate}"
            f"{stream_part}",
            progress=progress,
        )

    state["documents_size_bytes"] = (
        documents_path.stat().st_size if documents_path.exists() else 0
    )
    report_progress(None, 0, "resume" if state["documents_size_bytes"] else "start")
    with documents_path.open(mode) as handle:
        handle.seek(0, os.SEEK_END)
        for item in config.streams:
            stream = item.stream
            stream_bytes = state["stream_document_bytes"][stream.stream_id]
            stream_position = state["stream_positions"][stream.stream_id]
            stream_target = _effective_target(state, stream.stream_id)
            if stream_bytes >= stream_target:
                report_progress(stream, stream_bytes, "skip-completed-stream")
                continue
            report_progress(stream, stream_bytes, "start-stream")
            iterator = _iter_huggingface(stream)
            for _ in range(stream_position):
                next(iterator)
            while stream_bytes < _effective_target(state, stream.stream_id):
                try:
                    record = next(iterator)
                except StopIteration:
                    _checkpoint_state(state_path, handle, state)
                    allocations = _redistribute_exhausted_stream(
                        state,
                        config,
                        exhausted_item=item,
                        actual_bytes=stream_bytes,
                        stream_position=stream_position,
                    )
                    _checkpoint_state(state_path, handle, state)
                    allocation_summary = ", ".join(
                        f"{item['to_stream_id']}+={item['bytes']}"
                        for item in allocations
                    )
                    _emit_progress(
                        "[formal-space] stream-exhausted "
                        f"stream={stream.stream_id} "
                        f"actual={_format_gib(stream_bytes)} "
                        f"target={_format_gib(stream_target)} "
                        f"redistributed={allocation_summary}",
                        progress=progress,
                    )
                    break
                document, estimated_tokens = _document_from_record(
                    stream,
                    record,
                    stream_position,
                )
                encoded = f"{document.to_json_line()}\n".encode("utf-8")
                handle.write(encoded)
                stream_position += 1
                line_bytes = len(encoded)
                stream_bytes += line_bytes
                state["stream_positions"][stream.stream_id] = stream_position
                _update_counters(state, stream, document, line_bytes, estimated_tokens)
                records_since_checkpoint = (
                    state["records_written"] - last_checkpoint_record
                )
                bytes_since_checkpoint = handle.tell() - last_checkpoint_byte
                if (
                    records_since_checkpoint >= config.checkpoint_every_records
                    or bytes_since_checkpoint >= config.checkpoint_every_bytes
                ):
                    _checkpoint_state(state_path, handle, state)
                    report_progress(stream, stream_bytes, "checkpoint")
                    last_checkpoint_record = state["records_written"]
                    last_checkpoint_byte = state["documents_size_bytes"]
            _checkpoint_state(state_path, handle, state)
            report_progress(stream, stream_bytes, "finish-stream")
            last_checkpoint_record = state["records_written"]
            last_checkpoint_byte = state["documents_size_bytes"]

        _checkpoint_state(state_path, handle, state)
        report_progress(None, 0, "finish-writing")

    _emit_progress("[formal-space] hashing documents.jsonl ...", progress=progress)
    documents_sha256, line_count = _sha256_and_line_count(documents_path)
    if line_count != state["records_written"]:
        raise FormalSpaceError("documents line count does not match state")
    completed_distribution = _validate_completed_distribution(state, config, validation)
    manifest = {
        "schema_version": FORMAL_SPACE_SCHEMA_VERSION,
        "space_version": FORMAL_SPACE_VERSION,
        "plan_id": config.plan_id,
        "formal_training_eligible": True,
        "token_estimator": TOKEN_ESTIMATOR,
        "target_document_bytes": config.target_document_bytes,
        "actual_document_bytes": documents_path.stat().st_size,
        "actual_document_gib": round(documents_path.stat().st_size / GIB, 3),
        "record_count": state["records_written"],
        "estimated_tokens": state["estimated_tokens"],
        "documents_file": documents_path.name,
        "documents_sha256": documents_sha256,
        "language_counts": state["language_counts"],
        "content_counts": state["content_counts"],
        "source_counts": state["source_counts"],
        "privacy_warning_counts": state["privacy_warning_counts"],
        "estimated_tokens_by_language_bucket": state[
            "estimated_tokens_by_language_bucket"
        ],
        "estimated_tokens_by_content": state["estimated_tokens_by_content"],
        "estimated_tokens_by_source": state["estimated_tokens_by_source"],
        "source_byte_targets": validation["source_byte_targets"],
        "source_byte_fractions": validation["source_byte_fractions"],
        "source_target_ceiling_bytes": config.source_target_ceiling_bytes,
        "configured_exhaustion_fallback_stream_ids": list(
            config.exhaustion_fallback_stream_ids
        ),
        "effective_stream_target_bytes": state["stream_effective_target_bytes"],
        "exhausted_streams": state["exhausted_streams"],
        "redistributions": state["redistributions"],
        "checkpoint_every_records": config.checkpoint_every_records,
        "checkpoint_every_bytes": config.checkpoint_every_bytes,
        "configured_streams": [item.to_mapping() for item in config.streams],
        "completed_at": datetime.now(UTC).isoformat(),
        **completed_distribution,
    }
    _write_json_atomic(manifest_path, manifest)
    state["completed"] = True
    _write_json_atomic(state_path, state)
    _emit_progress(
        f"[formal-space] completed manifest={manifest_path.as_posix()}",
        progress=progress,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire or dry-run the formal data-space version."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the formal data-space plan without downloading data.",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip the free-space guard; intended only for tests or controlled runs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress output during acquisition.",
    )
    parser.add_argument(
        "--rebase-effective-targets",
        action="store_true",
        help=(
            "Replace resumable stream targets with this config's targets after a "
            "reviewed capacity replan; committed documents are never rewritten."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(".env", override=False)
    args = build_parser().parse_args(argv)
    if args.dry_run:
        summary = plan_formal_space(args.config, project_root=args.project_root)
    else:
        summary = acquire_formal_space(
            args.config,
            project_root=args.project_root,
            require_disk_space=not args.skip_disk_check,
            progress=not args.quiet,
            rebase_effective_targets=args.rebase_effective_targets,
        )
    print(
        json.dumps(
            {
                "actual_document_gib": summary.get("actual_document_gib"),
                "disk_space_ready": summary.get("disk_space_ready"),
                "formal_training_eligible": summary.get("formal_training_eligible"),
                "output_dir": summary.get("output_dir"),
                "plan_id": summary["plan_id"],
                "record_count": summary.get("record_count"),
                "target_document_gib": summary.get("target_document_gib"),
                "target_document_bytes": summary.get("target_document_bytes"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
