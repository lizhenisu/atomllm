"""Formal-data v0 streaming acquisition into canonical JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from atomllm.data.acquisition import classify_chinese_script
from atomllm.data.formal_plan import FormalDataPlan, load_formal_data_plan
from atomllm.data.mixture import load_pretraining_mixture
from atomllm.data.schema import SCHEMA_VERSION, CanonicalDocument, make_document_id


FORMAL_ACQUISITION_SCHEMA_VERSION = 1
ACQUISITION_VERSION = "formal-acquire-v0"
TOKEN_ESTIMATOR = "pretokenizer-char-v1"
_STREAM_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class FormalAcquisitionError(RuntimeError):
    """Raised when formal-data acquisition cannot safely continue."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormalAcquisitionError(f"{context} must be a mapping")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise FormalAcquisitionError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise FormalAcquisitionError(
            f"{context} has unknown fields: {', '.join(unknown)}"
        )


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalAcquisitionError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


def _safe_relative_path(value: Any, field_name: str) -> Path:
    path = Path(_string(value, field_name))
    if path.is_absolute() or ".." in path.parts:
        raise FormalAcquisitionError(f"{field_name} must be a safe relative path")
    return path


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise FormalAcquisitionError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise FormalAcquisitionError(f"{field_name} must be a non-negative integer")
    return value


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
        raise FormalAcquisitionError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalAcquisitionError(f"JSON file must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(
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
    os.replace(tmp, path)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def estimate_tokens(text: str, language: str) -> int:
    """Estimate tokens before the formal tokenizer exists.

    This is deliberately labeled as an estimate and must be recomputed after the
    stage-2 tokenizer is frozen.
    """
    if language in {"zh-Hans", "zh-Hant", "ja"}:
        return max(1, len(text))
    return max(1, len(text.encode("utf-8")) // 4)


def language_bucket(language: str) -> str:
    if language in {"zh-Hans", "zh-Hant", "en", "ja"}:
        return language
    return "other"


@dataclass(frozen=True, slots=True)
class StreamSpec:
    stream_id: str
    loader: str
    file_format: str | None
    data_file: str | None
    source_id: str
    dataset: str
    config_name: str
    split: str
    revision: str
    text_field: str
    id_field: str | None
    title_field: str | None
    url_field: str | None
    language: str
    content_type: str
    target_estimated_tokens: int
    initial_skip_records: int

    @classmethod
    def from_mapping(cls, value: Any) -> StreamSpec:
        data = _mapping(value, "stream")
        _exact_keys(
            data,
            {
                "stream_id",
                "loader",
                "file_format",
                "data_file",
                "source_id",
                "dataset",
                "config_name",
                "split",
                "revision",
                "text_field",
                "id_field",
                "title_field",
                "url_field",
                "language",
                "content_type",
                "target_estimated_tokens",
                "initial_skip_records",
            },
            "stream",
        )
        stream_id = _string(data["stream_id"], "stream_id")
        if _STREAM_ID_PATTERN.fullmatch(stream_id) is None:
            raise FormalAcquisitionError("stream_id must be lowercase path-safe")
        loader = _string(data["loader"], "loader")
        if loader not in {"hf_dataset", "hf_file"}:
            raise FormalAcquisitionError("loader must be hf_dataset or hf_file")
        file_format = _optional_string(data["file_format"], "file_format")
        data_file = _optional_string(data["data_file"], "data_file")
        if loader == "hf_dataset" and (
            file_format is not None or data_file is not None
        ):
            raise FormalAcquisitionError("hf_dataset streams must not set file fields")
        if loader == "hf_file":
            if file_format not in {"parquet", "json"}:
                raise FormalAcquisitionError("hf_file file_format must be parquet/json")
            if data_file is None:
                raise FormalAcquisitionError("hf_file streams must set data_file")
        return cls(
            stream_id=stream_id,
            loader=loader,
            file_format=file_format,
            data_file=data_file,
            source_id=_string(data["source_id"], "source_id"),
            dataset=_string(data["dataset"], "dataset"),
            config_name=_string(data["config_name"], "config_name"),
            split=_string(data["split"], "split"),
            revision=_string(data["revision"], "revision"),
            text_field=_string(data["text_field"], "text_field"),
            id_field=_optional_string(data["id_field"], "id_field"),
            title_field=_optional_string(data["title_field"], "title_field"),
            url_field=_optional_string(data["url_field"], "url_field"),
            language=_string(data["language"], "language"),
            content_type=_string(data["content_type"], "content_type"),
            target_estimated_tokens=_positive_int(
                data["target_estimated_tokens"], "target_estimated_tokens"
            ),
            initial_skip_records=_non_negative_int(
                data["initial_skip_records"], "initial_skip_records"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "loader": self.loader,
            "file_format": self.file_format,
            "data_file": self.data_file,
            "source_id": self.source_id,
            "dataset": self.dataset,
            "config_name": self.config_name,
            "split": self.split,
            "revision": self.revision,
            "text_field": self.text_field,
            "id_field": self.id_field,
            "title_field": self.title_field,
            "url_field": self.url_field,
            "language": self.language,
            "content_type": self.content_type,
            "target_estimated_tokens": self.target_estimated_tokens,
            "initial_skip_records": self.initial_skip_records,
        }


@dataclass(frozen=True, slots=True)
class FormalAcquisitionConfig:
    schema_version: int
    plan_id: str
    formal_plan_path: Path
    target_estimated_tokens: int
    token_estimator: str
    output_dir: Path
    streams: tuple[StreamSpec, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> FormalAcquisitionConfig:
        data = _mapping(value, "formal acquisition config")
        _exact_keys(
            data,
            {
                "schema_version",
                "plan_id",
                "formal_plan_path",
                "target_estimated_tokens",
                "token_estimator",
                "output_dir",
                "streams",
            },
            "formal acquisition config",
        )
        if data["schema_version"] != FORMAL_ACQUISITION_SCHEMA_VERSION:
            raise FormalAcquisitionError("schema_version must be 1")
        if data["token_estimator"] != TOKEN_ESTIMATOR:
            raise FormalAcquisitionError(f"token_estimator must be {TOKEN_ESTIMATOR}")
        streams_raw = data["streams"]
        if not isinstance(streams_raw, list) or not streams_raw:
            raise FormalAcquisitionError("streams must be a non-empty list")
        streams = tuple(StreamSpec.from_mapping(item) for item in streams_raw)
        stream_ids = [stream.stream_id for stream in streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise FormalAcquisitionError("stream_id values must be unique")
        target = _positive_int(
            data["target_estimated_tokens"], "target_estimated_tokens"
        )
        if sum(stream.target_estimated_tokens for stream in streams) != target:
            raise FormalAcquisitionError("stream targets must sum to target tokens")
        return cls(
            schema_version=FORMAL_ACQUISITION_SCHEMA_VERSION,
            plan_id=_string(data["plan_id"], "plan_id"),
            formal_plan_path=_safe_relative_path(
                data["formal_plan_path"], "formal_plan_path"
            ),
            target_estimated_tokens=target,
            token_estimator=TOKEN_ESTIMATOR,
            output_dir=_safe_relative_path(data["output_dir"], "output_dir"),
            streams=streams,
        )


def load_formal_acquisition_config(
    path: str | Path,
) -> FormalAcquisitionConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"formal acquisition config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise FormalAcquisitionError(
            f"invalid formal acquisition YAML: {error}"
        ) from error
    return FormalAcquisitionConfig.from_mapping(raw)


def _initial_state(
    config: FormalAcquisitionConfig,
    formal_plan: FormalDataPlan,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_ACQUISITION_SCHEMA_VERSION,
        "acquisition_version": ACQUISITION_VERSION,
        "plan_id": config.plan_id,
        "formal_data_plan_id": formal_plan.plan_id,
        "target_estimated_tokens": config.target_estimated_tokens,
        "token_estimator": config.token_estimator,
        "stream_positions": {
            stream.stream_id: stream.initial_skip_records for stream in config.streams
        },
        "stream_estimated_tokens": {stream.stream_id: 0 for stream in config.streams},
        "records_written": 0,
        "estimated_tokens": 0,
        "completed": False,
    }


def _load_or_create_state(
    output_dir: Path,
    config: FormalAcquisitionConfig,
    formal_plan: FormalDataPlan,
) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    documents_path = output_dir / "documents.jsonl"
    expected = _initial_state(config, formal_plan)
    if not state_path.exists():
        if documents_path.exists():
            raise FormalAcquisitionError(
                "documents.jsonl exists without state.json; refusing overwrite"
            )
        _write_json_atomic(state_path, expected)
        return expected
    state = _read_json(state_path)
    for key in (
        "schema_version",
        "acquisition_version",
        "plan_id",
        "formal_data_plan_id",
        "target_estimated_tokens",
        "token_estimator",
    ):
        if state.get(key) != expected[key]:
            raise FormalAcquisitionError(f"existing state mismatch: {key}")
    records_written = state.get("records_written")
    if type(records_written) is not int or records_written < 0:
        raise FormalAcquisitionError("state records_written is invalid")
    if _count_lines(documents_path) != records_written:
        raise FormalAcquisitionError("documents line count does not match state")
    return state


def _field(record: dict[str, Any], name: str | None) -> str | None:
    if name is None:
        return None
    value = record.get(name)
    if value is None:
        return None
    return str(value)


def _document_from_record(
    stream: StreamSpec,
    record: dict[str, Any],
    stream_position: int,
) -> tuple[CanonicalDocument, int]:
    text = record.get(stream.text_field)
    if not isinstance(text, str) or not text.strip():
        raise FormalAcquisitionError(f"stream {stream.stream_id} record has empty text")
    raw_record_id = _field(record, stream.id_field)
    if raw_record_id is None or not raw_record_id.strip():
        raw_record_id = f"position-{stream_position:09d}"
    source_record_id = f"{stream.stream_id}:{raw_record_id}"
    language = (
        classify_chinese_script(text)
        if stream.language == "auto-zh-script"
        else stream.language
    )
    title = _field(record, stream.title_field)
    url = _field(record, stream.url_field)
    estimated_tokens = estimate_tokens(text, language)
    document = CanonicalDocument.from_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "document_id": make_document_id(stream.source_id, source_record_id),
            "source_id": stream.source_id,
            "source_record_id": source_record_id,
            "text": text,
            "language": language,
            "content_type": stream.content_type,
            "privacy_warnings": [],
            "quality_warnings": [],
            "metadata": {
                "formal_v0_stream_id": stream.stream_id,
                "hf_dataset": stream.dataset,
                "hf_config_name": stream.config_name,
                "hf_revision": stream.revision,
                "hf_split": stream.split,
                "title": title,
                "url": url,
                "estimated_tokens": estimated_tokens,
                "token_estimator": TOKEN_ESTIMATOR,
            },
        }
    )
    return document, estimated_tokens


def _iter_huggingface(stream: StreamSpec):
    from datasets import load_dataset

    if stream.loader == "hf_file":
        data_file = (
            f"hf://datasets/{stream.dataset}@{stream.revision}/{stream.data_file}"
        )
        return iter(
            load_dataset(
                stream.file_format,
                data_files={stream.split: data_file},
                split=stream.split,
                streaming=True,
                token=True,
            )
        )
    return iter(
        load_dataset(
            stream.dataset,
            stream.config_name,
            split=stream.split,
            revision=stream.revision,
            streaming=True,
            token=True,
        )
    )


def acquire_formal_v0(
    config_path: str | Path = "configs/data/formal-v0-acquisition.yaml",
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root)
    config = load_formal_acquisition_config(root / config_path)
    formal_plan = load_formal_data_plan(
        root / config.formal_plan_path,
        project_root=root,
    )
    if not formal_plan.training_eligible:
        raise FormalAcquisitionError("formal data plan is not training_eligible")
    mixture = load_pretraining_mixture(root / "configs/data/pretraining-mixture.yaml")
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"
    state_path = output_dir / "state.json"
    state = _load_or_create_state(output_dir, config, formal_plan)
    if state.get("completed") is True:
        if not manifest_path.is_file():
            raise FormalAcquisitionError("completed state is missing manifest")
        return _read_json(manifest_path)

    mode = "a" if state["records_written"] else "w"
    with documents_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for stream in config.streams:
            stream_tokens = state["stream_estimated_tokens"][stream.stream_id]
            stream_position = state["stream_positions"][stream.stream_id]
            if stream_tokens >= stream.target_estimated_tokens:
                continue
            iterator = _iter_huggingface(stream)
            for _ in range(stream_position):
                next(iterator)
            while stream_tokens < stream.target_estimated_tokens:
                record = next(iterator)
                document, estimated_tokens = _document_from_record(
                    stream,
                    record,
                    stream_position,
                )
                handle.write(f"{document.to_json_line()}\n")
                handle.flush()
                os.fsync(handle.fileno())
                stream_position += 1
                stream_tokens += estimated_tokens
                state["stream_positions"][stream.stream_id] = stream_position
                state["stream_estimated_tokens"][stream.stream_id] = stream_tokens
                state["records_written"] += 1
                state["estimated_tokens"] += estimated_tokens
                _write_json_atomic(state_path, state)

    language_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    privacy_counts: Counter[str] = Counter()
    estimated_by_language: Counter[str] = Counter()
    estimated_by_language_bucket: Counter[str] = Counter()
    estimated_by_content: Counter[str] = Counter()
    estimated_by_source: Counter[str] = Counter()
    with documents_path.open(encoding="utf-8") as handle:
        for line in handle:
            document = CanonicalDocument.from_json_line(line)
            tokens = int(document.metadata["estimated_tokens"])
            language_counts[document.language] += 1
            content_counts[document.content_type] += 1
            source_counts[document.source_id] += 1
            privacy_counts.update(document.privacy_warnings)
            estimated_by_language[document.language] += tokens
            estimated_by_language_bucket[language_bucket(document.language)] += tokens
            estimated_by_content[document.content_type] += tokens
            estimated_by_source[document.source_id] += tokens

    covered_languages = set(estimated_by_language_bucket)
    covered_content = set(estimated_by_content)
    manifest = {
        "schema_version": FORMAL_ACQUISITION_SCHEMA_VERSION,
        "acquisition_version": ACQUISITION_VERSION,
        "plan_id": config.plan_id,
        "formal_data_plan_id": formal_plan.plan_id,
        "formal_training_eligible": True,
        "token_estimator": TOKEN_ESTIMATOR,
        "target_estimated_tokens": config.target_estimated_tokens,
        "estimated_tokens": state["estimated_tokens"],
        "record_count": state["records_written"],
        "documents_file": documents_path.name,
        "documents_bytes": documents_path.stat().st_size,
        "documents_sha256": _sha256(documents_path),
        "language_counts": dict(sorted(language_counts.items())),
        "content_counts": dict(sorted(content_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "estimated_tokens_by_language": dict(sorted(estimated_by_language.items())),
        "estimated_tokens_by_language_bucket": dict(
            sorted(estimated_by_language_bucket.items())
        ),
        "estimated_tokens_by_content": dict(sorted(estimated_by_content.items())),
        "estimated_tokens_by_source": dict(sorted(estimated_by_source.items())),
        "privacy_warning_counts": dict(sorted(privacy_counts.items())),
        "privacy_action": "warn",
        "configured_streams": [stream.to_mapping() for stream in config.streams],
        "uncovered_language_buckets": sorted(
            set(mixture.language_mix) - covered_languages
        ),
        "uncovered_content_buckets": sorted(set(mixture.content_mix) - covered_content),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(manifest_path, manifest)
    state["completed"] = True
    _write_json_atomic(state_path, state)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire the stage-1 formal-data v0 canonical JSONL artifact."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/formal-v0-acquisition.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(".env", override=False)
    args = build_parser().parse_args(argv)
    manifest = acquire_formal_v0(args.config, project_root=args.project_root)
    print(
        json.dumps(
            {
                "content_counts": manifest["content_counts"],
                "estimated_tokens": manifest["estimated_tokens"],
                "formal_training_eligible": manifest["formal_training_eligible"],
                "language_counts": manifest["language_counts"],
                "record_count": manifest["record_count"],
                "uncovered_content_buckets": manifest["uncovered_content_buckets"],
                "uncovered_language_buckets": manifest["uncovered_language_buckets"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
