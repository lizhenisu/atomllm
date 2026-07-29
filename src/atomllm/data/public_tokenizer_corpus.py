"""Build a resumable, public-only English/Simplified-Chinese tokenizer corpus."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sqlite3
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from atomllm.data.acquisition import (
    chinese_script_classifier_identity,
    classify_chinese_script,
)
from atomllm.data.schema import SCHEMA_VERSION, CanonicalDocument, make_document_id


SCHEMA = 1
DEFAULT_CONFIG = Path("configs/data/public-tokenizer-corpus-en-zh-v1.yaml")
ALLOWED_LANGUAGES = {"en", "zh-Hans", "code"}
ALLOWED_CONTENT_TYPES = {"general", "encyclopedia", "science", "math", "code"}
SOURCE_CACHE_ENV = "ATOMLLM_HF_SOURCE_CACHE"
_VERIFIED_SOURCE_CACHE_FILES: set[tuple[Path, int, int, str]] = set()


class PublicTokenizerCorpusError(RuntimeError):
    """Raised when the public tokenizer corpus contract is violated."""


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PublicTokenizerCorpusError(f"{field} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PublicTokenizerCorpusError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PublicTokenizerCorpusError(f"{field} must be a positive integer")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicTokenizerCorpusError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    dataset: str
    config_name: str | None
    split: str
    revision: str
    data_files_pattern: str | None
    file_format: str | None
    text_field: str
    id_field: str | None
    language: str
    content_type: str
    target_text_bytes: int
    minimum_characters: int
    maximum_characters: int
    score_field: str | None
    minimum_score: float | None
    exclude_if_true_fields: tuple[str, ...]
    license: str
    source_card: str
    text_conversion: str

    @classmethod
    def from_mapping(cls, value: Any) -> Source:
        required = {
            "source_id",
            "dataset",
            "config_name",
            "split",
            "revision",
            "data_files_pattern",
            "file_format",
            "text_field",
            "id_field",
            "language",
            "content_type",
            "target_text_bytes",
            "minimum_characters",
            "maximum_characters",
            "score_field",
            "minimum_score",
            "exclude_if_true_fields",
            "license",
            "source_card",
            "synthetic_content",
            "text_conversion",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise PublicTokenizerCorpusError(
                f"source fields must be exactly {sorted(required)}"
            )
        if value["synthetic_content"] is not False:
            raise PublicTokenizerCorpusError("synthetic training content is forbidden")
        if value["text_conversion"] not in {"none", "upstream-t2s"}:
            raise PublicTokenizerCorpusError(
                "text_conversion must be none or upstream-t2s; local conversion is forbidden"
            )
        revision = _string(value["revision"], "source.revision")
        if revision.lower() in {"main", "master", "latest", "head"}:
            raise PublicTokenizerCorpusError("source.revision must be immutable")
        language = _string(value["language"], "source.language")
        if language not in ALLOWED_LANGUAGES:
            raise PublicTokenizerCorpusError(
                f"source.language must be one of {sorted(ALLOWED_LANGUAGES)}"
            )
        content_type = _string(value["content_type"], "source.content_type")
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise PublicTokenizerCorpusError("unsupported source.content_type")
        if (language == "code") != (content_type == "code"):
            raise PublicTokenizerCorpusError(
                "code language and content type must match"
            )
        config_name = value["config_name"]
        pattern = value["data_files_pattern"]
        file_format = value["file_format"]
        if config_name is not None and not isinstance(config_name, str):
            raise PublicTokenizerCorpusError("source.config_name must be string/null")
        if pattern is not None and not isinstance(pattern, str):
            raise PublicTokenizerCorpusError(
                "source.data_files_pattern must be string/null"
            )
        if file_format not in {None, "json", "parquet"}:
            raise PublicTokenizerCorpusError(
                "source.file_format must be json, parquet, or null"
            )
        if (pattern is None) != (file_format is None):
            raise PublicTokenizerCorpusError(
                "source.data_files_pattern and file_format must be set together"
            )
        if config_name is not None and pattern is not None:
            raise PublicTokenizerCorpusError(
                "source cannot combine config_name and data_files_pattern"
            )
        id_field = value["id_field"]
        score_field = value["score_field"]
        for item, field in ((id_field, "id_field"), (score_field, "score_field")):
            if item is not None and not isinstance(item, str):
                raise PublicTokenizerCorpusError(f"source.{field} must be string/null")
        minimum_score = value["minimum_score"]
        if minimum_score is not None and type(minimum_score) not in {int, float}:
            raise PublicTokenizerCorpusError("source.minimum_score must be number/null")
        if (score_field is None) != (minimum_score is None):
            raise PublicTokenizerCorpusError(
                "source score_field and minimum_score must be set together"
            )
        minimum = _positive_int(value["minimum_characters"], "minimum_characters")
        maximum = _positive_int(value["maximum_characters"], "maximum_characters")
        if minimum > maximum:
            raise PublicTokenizerCorpusError("source character bounds are invalid")
        excluded = value["exclude_if_true_fields"]
        if not isinstance(excluded, list) or not all(
            isinstance(item, str) and item for item in excluded
        ):
            raise PublicTokenizerCorpusError(
                "source.exclude_if_true_fields must be a string list"
            )
        return cls(
            source_id=_string(value["source_id"], "source.source_id"),
            dataset=_string(value["dataset"], "source.dataset"),
            config_name=config_name,
            split=_string(value["split"], "source.split"),
            revision=revision,
            data_files_pattern=pattern,
            file_format=file_format,
            text_field=_string(value["text_field"], "source.text_field"),
            id_field=id_field,
            language=language,
            content_type=content_type,
            target_text_bytes=_positive_int(
                value["target_text_bytes"], "source.target_text_bytes"
            ),
            minimum_characters=minimum,
            maximum_characters=maximum,
            score_field=score_field,
            minimum_score=None if minimum_score is None else float(minimum_score),
            exclude_if_true_fields=tuple(excluded),
            license=_string(value["license"], "source.license"),
            source_card=_string(value["source_card"], "source.source_card"),
            text_conversion=value["text_conversion"],
        )


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    output_dir: Path
    checkpoint_every_records: int
    checkpoint_every_bytes: int
    minimum_free_bytes: int
    sources: tuple[Source, ...]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "name",
        "output_dir",
        "checkpoint_every_records",
        "checkpoint_every_bytes",
        "minimum_free_bytes",
        "sources",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise PublicTokenizerCorpusError(f"config fields must be {sorted(required)}")
    if raw["schema_version"] != SCHEMA:
        raise PublicTokenizerCorpusError(f"schema_version must be {SCHEMA}")
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise PublicTokenizerCorpusError("sources must be a non-empty list")
    sources = tuple(Source.from_mapping(item) for item in raw["sources"])
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise PublicTokenizerCorpusError("source_id values must be unique")
    language_targets = Counter()
    for source in sources:
        language_targets[source.language] += source.target_text_bytes
    if set(language_targets) != ALLOWED_LANGUAGES:
        raise PublicTokenizerCorpusError("config must contain en, zh-Hans, and code")
    total = sum(language_targets.values())
    if not (
        language_targets["en"] * 2 == total
        and language_targets["code"] * 10 == total
        and language_targets["zh-Hans"] * 5 == total * 2
    ):
        raise PublicTokenizerCorpusError(
            "targets must be exactly 50% English, 10% code, and 40% Chinese"
        )
    return Config(
        name=_string(raw["name"], "name"),
        output_dir=_safe_path(raw["output_dir"], "output_dir"),
        checkpoint_every_records=_positive_int(
            raw["checkpoint_every_records"], "checkpoint_every_records"
        ),
        checkpoint_every_bytes=_positive_int(
            raw["checkpoint_every_bytes"], "checkpoint_every_bytes"
        ),
        minimum_free_bytes=_positive_int(
            raw["minimum_free_bytes"], "minimum_free_bytes"
        ),
        sources=sources,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_cache_repository(source: Source) -> Path | None:
    raw = os.environ.get(SOURCE_CACHE_ENV)
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    repository = (root / source.dataset / source.revision).resolve()
    if not repository.is_relative_to(root):
        raise PublicTokenizerCorpusError("source cache repository escapes its root")
    return repository


def _cached_source_files(
    source: Source, remote_files: Iterable[str]
) -> tuple[list[str], int]:
    """Replace verified Hub files with exact local copies in the same order."""
    files = [str(path) for path in remote_files]
    repository = _source_cache_repository(source)
    if repository is None:
        return files, 0
    prefix = f"hf://datasets/{source.dataset}@{source.revision}/"
    replacements = 0
    resolved_files: list[str] = []
    for remote in files:
        if not remote.startswith(prefix):
            resolved_files.append(remote)
            continue
        relative = Path(remote.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise PublicTokenizerCorpusError("source cache file path is unsafe")
        local = (repository / relative).resolve()
        metadata = (
            repository
            / ".cache/huggingface/download"
            / relative.parent
            / f"{relative.name}.metadata"
        ).resolve()
        # Hugging Face creates cache bookkeeping while a download is still in
        # progress. Only replace a remote URI once both final files exist;
        # otherwise the streaming reader safely keeps using the remote URI.
        if not local.is_file() or not metadata.is_file():
            resolved_files.append(remote)
            continue
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != source.revision:
            raise PublicTokenizerCorpusError(
                f"source cache revision mismatch: {relative.as_posix()}"
            )
        expected_sha256 = lines[1]
        if len(expected_sha256) != 64:
            raise PublicTokenizerCorpusError(
                f"source cache SHA-256 metadata is invalid: {relative.as_posix()}"
            )
        stat = local.stat()
        identity = (local, stat.st_size, stat.st_mtime_ns, expected_sha256)
        if identity not in _VERIFIED_SOURCE_CACHE_FILES:
            if _sha256(local) != expected_sha256:
                raise PublicTokenizerCorpusError(
                    f"source cache SHA-256 mismatch: {relative.as_posix()}"
                )
            _VERIFIED_SOURCE_CACHE_FILES.add(identity)
        resolved_files.append(str(local))
        replacements += 1
    return resolved_files, replacements


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_source_dataset(source: Source) -> Any:
    from datasets import load_dataset, load_dataset_builder

    if source.data_files_pattern is None:
        if _source_cache_repository(source) is not None:
            builder = load_dataset_builder(
                source.dataset,
                source.config_name,
                revision=source.revision,
                token=True,
            )
            data_files = builder.config.data_files
            if data_files is None or source.split not in data_files:
                raise PublicTokenizerCorpusError(
                    f"cannot resolve source files for cache: {source.source_id}"
                )
            files, replacements = _cached_source_files(source, data_files[source.split])
            if replacements:
                print(
                    f"[tokenizer-corpus] source cache files={replacements} "
                    f"source={source.source_id}",
                    flush=True,
                )
                return load_dataset(
                    source.dataset,
                    source.config_name,
                    data_files={source.split: files},
                    split=source.split,
                    revision=source.revision,
                    streaming=True,
                    token=True,
                )
        return load_dataset(
            source.dataset,
            source.config_name,
            split=source.split,
            revision=source.revision,
            streaming=True,
            token=True,
        )
    else:
        from huggingface_hub import HfApi

        files = sorted(
            name
            for name in HfApi().list_repo_files(
                source.dataset,
                repo_type="dataset",
                revision=source.revision,
                token=True,
            )
            if fnmatch.fnmatch(name, source.data_files_pattern)
        )
        if not files:
            raise PublicTokenizerCorpusError(
                f"no files match {source.source_id}: {source.data_files_pattern}"
            )
        urls = [
            f"hf://datasets/{source.dataset}@{source.revision}/{name}" for name in files
        ]
        urls, replacements = _cached_source_files(source, urls)
        if replacements:
            print(
                f"[tokenizer-corpus] source cache files={replacements} "
                f"source={source.source_id}",
                flush=True,
            )
        return load_dataset(
            source.file_format,
            data_files={source.split: urls},
            split=source.split,
            streaming=True,
            token=True,
        )


def _resume_source_dataset(
    source: Source,
    *,
    records_seen: int,
    iterator_checkpoint: Mapping[str, Any] | None,
) -> Any:
    dataset = _load_source_dataset(source)
    if iterator_checkpoint is None:
        return dataset.skip(records_seen)
    if iterator_checkpoint.get("records_seen") != records_seen:
        raise PublicTokenizerCorpusError(
            f"iterator checkpoint mismatch for {source.source_id}"
        )
    dataset_state = iterator_checkpoint.get("dataset_state")
    if not isinstance(dataset_state, dict):
        raise PublicTokenizerCorpusError(
            f"iterator checkpoint is invalid for {source.source_id}"
        )
    base_skip_records = iterator_checkpoint.get("base_skip_records")
    if type(base_skip_records) is not int or not 0 <= base_skip_records <= records_seen:
        raise PublicTokenizerCorpusError(
            f"iterator checkpoint base is invalid for {source.source_id}"
        )
    resumed = dataset.skip(base_skip_records)
    resumed.load_state_dict(dataset_state)
    _bind_live_resumed_state_dict(resumed)
    return resumed


def _bind_live_resumed_state_dict(dataset: Any) -> None:
    """Keep ``IterableDataset.state_dict`` live after loading a checkpoint.

    The locked ``datasets`` release initializes the prepared iterable's state,
    then ``load_state_dict`` initializes it a second time.  The public dataset
    object retains references from the first initialization, so its reported
    state remains frozen at the loaded checkpoint while iteration advances.
    Rebind that public state to the prepared iterable immediately after each
    preparation.  This is intentionally limited to resumed streams; fresh
    streams already expose live state correctly.
    """
    prepare = getattr(dataset, "_prepare_ex_iterable_for_iteration", None)
    if not callable(prepare):
        return

    def prepare_with_live_state(*args: Any, **kwargs: Any) -> Any:
        prepared = prepare(*args, **kwargs)
        prepared_state = getattr(prepared, "_state_dict", None)
        epoch = getattr(dataset, "epoch", None)
        if isinstance(prepared_state, dict) and type(epoch) is int:
            dataset._state_dict = {
                "examples_iterable": prepared_state,
                "epoch": epoch,
            }
        return prepared

    dataset._prepare_ex_iterable_for_iteration = prepare_with_live_state


def _dataset_state_cursor(value: Any) -> tuple[int, int, int, int, int]:
    """Return a comparable best-effort cursor from a datasets state tree.

    ``datasets`` stores wrapper offsets (for example ``skipped``) and the
    underlying Arrow/rebatch positions in different nested mappings.  Taking
    the lexicographic maximum of one mapping at a time loses the underlying
    progress whenever a large, constant wrapper offset is present.  Merge the
    maximum value for each cursor component instead so an advancing shard or
    batch remains observable after ``Dataset.skip``.
    """
    maximums = [0, 0, 0, 0, 0]

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            batch_position = item.get("batch_idx")
            if type(batch_position) is not int:
                batch_position = item.get("position")
            cursor = (
                field if type(field) is int and field >= 0 else 0
                for field in (
                    item.get("skipped"),
                    item.get("shard_idx"),
                    item.get("shard_example_idx"),
                    batch_position,
                    item.get("num_chunks_since_previous_state"),
                )
            )
            for index, field in enumerate(cursor):
                maximums[index] = max(maximums[index], field)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(maximums)


def _usable_iterator_checkpoint(
    checkpoint: Mapping[str, Any] | None, records_seen: int
) -> Mapping[str, Any] | None:
    if checkpoint is None:
        return None
    base_skip = checkpoint.get("base_skip_records")
    dataset_state = checkpoint.get("dataset_state")
    state_records = checkpoint.get("dataset_state_records_seen", records_seen)
    if (
        type(base_skip) is not int
        or type(state_records) is not int
        or not 0 <= base_skip <= state_records <= records_seen
        or not isinstance(dataset_state, dict)
    ):
        return None
    cursor = _dataset_state_cursor(dataset_state)
    if state_records > base_skip and not any(cursor[1:]):
        # A SkipExamplesIterable can retain only its constant wrapper offset
        # while the underlying Arrow/rebatch cursor resets to zero.  That does
        # not prove progress beyond ``base_skip`` and must be safely rebased.
        return None
    return checkpoint


def _iter_source(source: Source, skip: int) -> Iterable[Mapping[str, Any]]:
    return _resume_source_dataset(
        source,
        records_seen=skip,
        iterator_checkpoint=None,
    )


def _accepted_text(source: Source, record: Mapping[str, Any]) -> tuple[str, str] | None:
    text = record.get(source.text_field)
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not source.minimum_characters <= len(text) <= source.maximum_characters:
        return None
    if any(record.get(field) is True for field in source.exclude_if_true_fields):
        return None
    if source.score_field is not None:
        try:
            score = float(record[source.score_field])
        except KeyError, TypeError, ValueError:
            return None
        if score < source.minimum_score:  # type: ignore[operator]
            return None
    if source.language == "zh-Hans" and classify_chinese_script(text) != "zh-Hans":
        return None
    raw_id = record.get(source.id_field) if source.id_field is not None else None
    return text, "" if raw_id is None else str(raw_id)


_ACCEPTANCE_WORKER_SOURCE: Source | None = None


def _initialize_acceptance_worker(source: Source) -> None:
    global _ACCEPTANCE_WORKER_SOURCE
    _ACCEPTANCE_WORKER_SOURCE = source


def _accepted_text_worker(record: Mapping[str, Any]) -> tuple[str, str] | None:
    if _ACCEPTANCE_WORKER_SOURCE is None:
        raise RuntimeError("acceptance worker was not initialized")
    return _accepted_text(_ACCEPTANCE_WORKER_SOURCE, record)


def _open_fingerprints(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS fingerprints (digest BLOB PRIMARY KEY) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    return connection


def _initial_state(config: Config, config_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "name": config.name,
        "config_sha256": config_sha,
        "committed_output_bytes": 0,
        "records_written": 0,
        "source_records_seen": {source.source_id: 0 for source in config.sources},
        "source_text_bytes": {source.source_id: 0 for source in config.sources},
        "source_documents": {source.source_id: 0 for source in config.sources},
        "source_iterator_states": {},
        "duplicate_documents": 0,
        "rejected_documents": 0,
        "completed": False,
    }


def _restore_output(output: Path, state: dict[str, Any]) -> None:
    documents = output / "documents.jsonl"
    committed = state["committed_output_bytes"]
    if documents.exists():
        with documents.open("ab") as handle:
            handle.truncate(committed)
    elif committed:
        raise PublicTokenizerCorpusError("committed tokenizer corpus is missing")


def _fingerprint_metadata(
    connection: sqlite3.Connection, *, committed_output_bytes: int, records: int
) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
        (
            ("committed_output_bytes", str(committed_output_bytes)),
            ("records", str(records)),
        ),
    )


def _content_bound_source_record_id(raw_id: str, digest: bytes) -> str:
    """Disambiguate upstream record IDs that are reused for different content."""
    suffix = f"#sha256-{digest.hex()}"
    return raw_id if raw_id.endswith(suffix) else f"{raw_id}{suffix}"


def _rebuild_fingerprints(
    documents: Path,
    database: Path,
    *,
    committed_output_bytes: int,
    records: int,
) -> sqlite3.Connection:
    if database.is_file():
        connection = _open_fingerprints(database)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        fingerprint_count = connection.execute(
            "SELECT COUNT(*) FROM fingerprints"
        ).fetchone()[0]
        if (
            metadata
            == {
                "committed_output_bytes": str(committed_output_bytes),
                "records": str(records),
            }
            and fingerprint_count == records
        ):
            print(
                f"[tokenizer-corpus] reuse fingerprints records={records}",
                flush=True,
            )
            return connection
        connection.close()
    database.unlink(missing_ok=True)
    database.with_suffix(f"{database.suffix}-wal").unlink(missing_ok=True)
    database.with_suffix(f"{database.suffix}-shm").unlink(missing_ok=True)
    connection = _open_fingerprints(database)
    observed_records = 0
    if documents.exists():
        batch: list[tuple[bytes]] = []
        with documents.open(encoding="utf-8") as handle:
            for line in handle:
                observed_records += 1
                text = CanonicalDocument.from_json_line(line).text
                batch.append((hashlib.sha256(text.encode("utf-8")).digest(),))
                if len(batch) == 10_000:
                    connection.executemany(
                        "INSERT OR IGNORE INTO fingerprints VALUES (?)", batch
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO fingerprints VALUES (?)", batch
            )
    if observed_records != records:
        connection.close()
        raise PublicTokenizerCorpusError(
            "committed document count does not match resume state"
        )
    fingerprint_count = connection.execute(
        "SELECT COUNT(*) FROM fingerprints"
    ).fetchone()[0]
    if fingerprint_count != records:
        connection.close()
        raise PublicTokenizerCorpusError(
            "committed documents contain duplicate fingerprints"
        )
    _fingerprint_metadata(
        connection,
        committed_output_bytes=committed_output_bytes,
        records=records,
    )
    connection.commit()
    return connection


_TRANSIENT_SOURCE_ERROR_TEXT = (
    "client has been closed",
    "read operation timed out",
    "read timed out",
    "connect timeout",
    "connection reset",
    "connection aborted",
    "remote protocol error",
    "server disconnected",
    "temporary failure in name resolution",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)


def _is_transient_source_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        module = type(current).__module__
        name = type(current).__name__
        if module.startswith(("httpx", "httpcore")) and name in {
            "ConnectError",
            "ConnectTimeout",
            "NetworkError",
            "ProtocolError",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "TimeoutException",
            "WriteError",
        }:
            return True
        message = str(current).lower()
        if any(fragment in message for fragment in _TRANSIENT_SOURCE_ERROR_TEXT):
            return True
        current = current.__cause__ or current.__context__
    return False


def _reset_huggingface_http_client() -> None:
    """Discard poisoned Hub clients and cached filesystems before retrying."""
    try:
        from huggingface_hub import HfFileSystem
        from huggingface_hub.utils import close_session

        close_session()
        # HfFileSystem is cached by fsspec. Closing huggingface_hub's global
        # HTTP client without evicting these instances makes the next dataset
        # attempt reuse a filesystem backed by that closed client, producing
        # an endless "client has been closed" retry loop.
        HfFileSystem.clear_instance_cache()
    except Exception as error:
        print(
            "[tokenizer-corpus] warning: failed to reset Hugging Face clients: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )


def build_with_retries(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
    classification_workers: int = 1,
    maximum_source_restarts: int = 1000,
) -> dict[str, Any]:
    if type(maximum_source_restarts) is not int or maximum_source_restarts < 0:
        raise PublicTokenizerCorpusError(
            "maximum_source_restarts must be a non-negative integer"
        )
    restarts = 0
    while True:
        try:
            return build(
                config_path,
                project_root=project_root,
                classification_workers=classification_workers,
            )
        except Exception as error:
            if (
                not _is_transient_source_error(error)
                or restarts >= maximum_source_restarts
            ):
                raise
            restarts += 1
            delay = min(60, 5 * 2 ** min(restarts - 1, 4))
            _reset_huggingface_http_client()
            print(
                "[tokenizer-corpus] transient source failure; "
                f"restart={restarts}/{maximum_source_restarts} "
                f"delay_seconds={delay} error={type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(delay)


def build(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
    classification_workers: int = 1,
) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    if not 1 <= classification_workers <= cpu_count:
        raise PublicTokenizerCorpusError(
            f"classification_workers must be in [1, {cpu_count}]"
        )
    root = Path(project_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = load_config(path)
    config_sha = _sha256(path)
    output = root / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (
        os.statvfs(output).f_bavail * os.statvfs(output).f_frsize
        < config.minimum_free_bytes
    ):
        raise PublicTokenizerCorpusError("insufficient free space for tokenizer corpus")
    state_path = output / "state.json"
    documents = output / "documents.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha:
            raise PublicTokenizerCorpusError("resume config SHA-256 mismatch")
    else:
        if documents.exists():
            raise PublicTokenizerCorpusError("documents exist without resume state")
        state = _initial_state(config, config_sha)
        _write_json(state_path, state)
    state.setdefault("source_iterator_states", {})
    classifier_identity = chinese_script_classifier_identity()
    existing_classifier = state.get("chinese_script_classifier")
    if existing_classifier is not None and existing_classifier != classifier_identity:
        raise PublicTokenizerCorpusError("Chinese script classifier identity mismatch")
    state["chinese_script_classifier"] = classifier_identity
    if state["completed"]:
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    _restore_output(output, state)
    fingerprints = _rebuild_fingerprints(
        documents,
        output / "fingerprints.sqlite3",
        committed_output_bytes=state["committed_output_bytes"],
        records=state["records_written"],
    )
    since_records = 0
    since_bytes = 0
    current_dataset = None
    current_source_id: str | None = None
    current_base_skip_records = 0
    current_dataset_state: dict[str, Any] | None = None
    current_dataset_state_records_seen = 0
    current_dataset_state_cursor = (0, 0, 0, 0, 0)

    def checkpoint(handle) -> None:
        nonlocal since_records, since_bytes
        nonlocal current_dataset_state, current_dataset_state_records_seen
        nonlocal current_dataset_state_cursor
        handle.flush()
        os.fsync(handle.fileno())
        _fingerprint_metadata(
            fingerprints,
            committed_output_bytes=handle.tell(),
            records=state["records_written"],
        )
        fingerprints.commit()
        if current_dataset is not None and current_source_id is not None:
            candidate_state = current_dataset.state_dict()
            candidate_cursor = _dataset_state_cursor(candidate_state)
            committed_seen = state["source_records_seen"][current_source_id]
            if candidate_cursor > current_dataset_state_cursor:
                current_dataset_state = candidate_state
                current_dataset_state_records_seen = committed_seen
                current_dataset_state_cursor = candidate_cursor
            if current_dataset_state is None:
                state["source_iterator_states"].pop(current_source_id, None)
            else:
                state["source_iterator_states"][current_source_id] = {
                    "records_seen": committed_seen,
                    "base_skip_records": current_base_skip_records,
                    "dataset_state_records_seen": (current_dataset_state_records_seen),
                    "replay_through_records_seen": committed_seen,
                    "dataset_state": current_dataset_state,
                }
        state["committed_output_bytes"] = handle.tell()
        _write_json(state_path, state)
        since_records = 0
        since_bytes = 0
        print(
            f"[tokenizer-corpus] documents={state['records_written']} "
            f"text_bytes={sum(state['source_text_bytes'].values())} "
            f"source={current_source_id} "
            f"source_records_seen={state['source_records_seen'].get(current_source_id, 0)} "
            f"iterator_state_records_seen={current_dataset_state_records_seen}",
            flush=True,
        )

    try:
        with documents.open("a", encoding="utf-8", newline="\n") as handle:
            for source in config.sources:
                current_dataset = None
                current_source_id = source.source_id
                seen = state["source_records_seen"][source.source_id]
                committed_seen = seen
                target = source.target_text_bytes
                if state["source_text_bytes"][source.source_id] >= target:
                    continue
                raw_iterator_checkpoint = state["source_iterator_states"].get(
                    source.source_id
                )
                iterator_checkpoint = _usable_iterator_checkpoint(
                    raw_iterator_checkpoint, seen
                )
                if iterator_checkpoint is None:
                    if raw_iterator_checkpoint is not None:
                        print(
                            "[tokenizer-corpus] invalid iterator checkpoint; "
                            f"safe_rebase_source={source.source_id} "
                            f"records_seen={seen}",
                            flush=True,
                        )
                    state["source_iterator_states"].pop(source.source_id, None)
                    _write_json(state_path, state)
                current_base_skip_records = (
                    seen
                    if iterator_checkpoint is None
                    else iterator_checkpoint["base_skip_records"]
                )
                current_dataset_state = (
                    None
                    if iterator_checkpoint is None
                    else iterator_checkpoint["dataset_state"]
                )
                current_dataset_state_records_seen = (
                    seen
                    if iterator_checkpoint is None
                    else iterator_checkpoint.get(
                        "dataset_state_records_seen",
                        iterator_checkpoint["records_seen"],
                    )
                )
                current_dataset_state_cursor = _dataset_state_cursor(
                    current_dataset_state
                )
                current_dataset = _resume_source_dataset(
                    source,
                    records_seen=seen,
                    iterator_checkpoint=iterator_checkpoint,
                )
                source_iterator = iter(current_dataset)
                iterator_position = current_dataset_state_records_seen
                batch_size = max(256, classification_workers * 16)
                executor = (
                    ProcessPoolExecutor(
                        max_workers=classification_workers,
                        mp_context=get_context("spawn"),
                        initializer=_initialize_acceptance_worker,
                        initargs=(source,),
                    )
                    if source.language == "zh-Hans" and classification_workers > 1
                    else None
                )
                try:
                    while state["source_text_bytes"][source.source_id] < target:
                        batch_start_position = iterator_position
                        records = list(islice(source_iterator, batch_size))
                        if not records:
                            raise PublicTokenizerCorpusError(
                                f"source exhausted before target: {source.source_id}"
                            )
                        accepted_records = (
                            executor.map(_accepted_text_worker, records, chunksize=8)
                            if executor is not None
                            else (_accepted_text(source, record) for record in records)
                        )
                        target_reached = False
                        for record, accepted in zip(
                            records, accepted_records, strict=True
                        ):
                            iterator_position += 1
                            is_resume_replay = iterator_position <= committed_seen
                            seen = max(seen, iterator_position)
                            state["source_records_seen"][source.source_id] = seen
                            if is_resume_replay and iterator_position == committed_seen:
                                print(
                                    "[tokenizer-corpus] resume replay verified "
                                    f"source={source.source_id} "
                                    f"anchor_records_seen="
                                    f"{current_dataset_state_records_seen} "
                                    f"replay_through_records_seen={committed_seen}",
                                    flush=True,
                                )
                            if accepted is None:
                                if not is_resume_replay:
                                    state["rejected_documents"] += 1
                                continue
                            text, raw_id = accepted
                            digest = hashlib.sha256(text.encode("utf-8")).digest()
                            if is_resume_replay:
                                if (
                                    fingerprints.execute(
                                        "SELECT 1 FROM fingerprints WHERE digest = ?",
                                        (digest,),
                                    ).fetchone()
                                    is None
                                ):
                                    raise PublicTokenizerCorpusError(
                                        "resume replay fingerprint is missing for "
                                        f"{source.source_id}"
                                    )
                                continue
                            if (
                                fingerprints.execute(
                                    "INSERT OR IGNORE INTO fingerprints VALUES (?)",
                                    (digest,),
                                ).rowcount
                                == 0
                            ):
                                state["duplicate_documents"] += 1
                                continue
                            upstream_record_id = raw_id or f"position-{seen - 1:012d}"
                            source_record_id = _content_bound_source_record_id(
                                upstream_record_id, digest
                            )
                            document = CanonicalDocument.from_mapping(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "document_id": make_document_id(
                                        source.source_id, source_record_id
                                    ),
                                    "source_id": source.source_id,
                                    "source_record_id": source_record_id,
                                    "text": text,
                                    "language": source.language,
                                    "content_type": source.content_type,
                                    "privacy_warnings": [],
                                    "quality_warnings": [],
                                    "metadata": {
                                        "dataset": source.dataset,
                                        "revision": source.revision,
                                        "license": source.license,
                                        "source_card": source.source_card,
                                        "simplified_chinese_only": (
                                            source.language == "zh-Hans"
                                        ),
                                        "upstream_text_conversion": (
                                            "none"
                                            if source.language != "zh-Hans"
                                            else source.text_conversion
                                        ),
                                        "local_text_conversion": "none",
                                        "upstream_quality_score": (
                                            None
                                            if source.score_field is None
                                            else float(record[source.score_field])
                                        ),
                                        "upstream_quality_score_field": (
                                            source.score_field
                                        ),
                                        "unicode_normalization_for_dedup": (
                                            unicodedata.normalize("NFC", text) == text
                                        ),
                                    },
                                }
                            )
                            line = document.to_json_line() + "\n"
                            handle.write(line)
                            text_bytes = len(text.encode("utf-8"))
                            state["records_written"] += 1
                            state["source_documents"][source.source_id] += 1
                            state["source_text_bytes"][source.source_id] += text_bytes
                            since_records += 1
                            since_bytes += len(line.encode("utf-8"))
                            if state["source_text_bytes"][source.source_id] >= target:
                                target_reached = True
                                break
                        if target_reached:
                            # The streaming iterator has consumed the full batch. The
                            # ignored tail cannot affect a source whose target is done.
                            iterator_position = batch_start_position + len(records)
                            seen = max(seen, iterator_position)
                            state["source_records_seen"][source.source_id] = seen
                        if (
                            target_reached
                            or since_records >= config.checkpoint_every_records
                            or since_bytes >= config.checkpoint_every_bytes
                        ):
                            checkpoint(handle)
                        if target_reached:
                            break
                finally:
                    if executor is not None:
                        executor.shutdown(cancel_futures=True)
                    close_iterator = getattr(source_iterator, "close", None)
                    if callable(close_iterator):
                        close_iterator()
            checkpoint(handle)
    finally:
        fingerprints.close()
    language_text_bytes = Counter()
    content_text_bytes = Counter()
    by_id = {source.source_id: source for source in config.sources}
    for source_id, count in state["source_text_bytes"].items():
        language_text_bytes[by_id[source_id].language] += count
        content_text_bytes[by_id[source_id].content_type] += count
    manifest = {
        "schema_version": SCHEMA,
        "name": config.name,
        "config_sha256": config_sha,
        "config": {"name": "config.yaml", "sha256": config_sha},
        "document_count": state["records_written"],
        "text_bytes": sum(state["source_text_bytes"].values()),
        "language_text_bytes": dict(sorted(language_text_bytes.items())),
        "content_text_bytes": dict(sorted(content_text_bytes.items())),
        "source_text_bytes": state["source_text_bytes"],
        "source_documents": state["source_documents"],
        "duplicate_documents": state["duplicate_documents"],
        "rejected_documents": state["rejected_documents"],
        "classification_workers": classification_workers,
        "chinese_script_classifier": classifier_identity,
        "documents": {
            "name": documents.name,
            "size_bytes": documents.stat().st_size,
            "sha256": _sha256(documents),
        },
        "language_contract": {
            "training_ratio": "en:code:zh-Hans=50:10:40",
            "code_is_separate_from_english": True,
            "simplified_chinese_only": True,
            "upstream_t2s_allowed_when_declared": True,
            "local_text_conversion": "none",
            "privacy_filtering": "none",
            "code_is_a_content_axis": True,
        },
        "synthetic_training_content": False,
        "sources": [
            {
                "source_id": source.source_id,
                "dataset": source.dataset,
                "revision": source.revision,
                "language": source.language,
                "content_type": source.content_type,
                "target_text_bytes": source.target_text_bytes,
                "minimum_characters": source.minimum_characters,
                "maximum_characters": source.maximum_characters,
                "score_field": source.score_field,
                "minimum_score": source.minimum_score,
                "exclude_if_true_fields": list(source.exclude_if_true_fields),
                "data_files_pattern": source.data_files_pattern,
                "file_format": source.file_format,
                "license": source.license,
                "source_card": source.source_card,
                "text_conversion": source.text_conversion,
                "synthetic_content": False,
            }
            for source in config.sources
        ],
    }
    shutil.copyfile(path, output / "config.yaml")
    _write_json(output / "manifest.json", manifest)
    (output / "COMPLETED").write_text(
        f"{_sha256(output / 'manifest.json')}  manifest.json\n", encoding="utf-8"
    )
    state["completed"] = True
    _write_json(state_path, state)
    return manifest


def probe_sources(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    maximum_records_per_source: int = 1000,
) -> dict[str, Any]:
    """Read remote rows without writing and prove every source can emit data."""
    if maximum_records_per_source <= 0:
        raise PublicTokenizerCorpusError("maximum_records_per_source must be positive")
    config = load_config(config_path)
    results = []
    for source in config.sources:
        seen = 0
        accepted = None
        for record in _iter_source(source, 0):
            seen += 1
            accepted = _accepted_text(source, record)
            if accepted is not None or seen >= maximum_records_per_source:
                break
        if accepted is None:
            raise PublicTokenizerCorpusError(
                f"source emitted no accepted probe record: {source.source_id}"
            )
        text, raw_id = accepted
        results.append(
            {
                "source_id": source.source_id,
                "records_examined": seen,
                "accepted_text_bytes": len(text.encode("utf-8")),
                "stable_id_available": bool(raw_id),
                "language": source.language,
                "content_type": source.content_type,
            }
        )
    return {
        "name": config.name,
        "text_conversion": "none",
        "synthetic_training_content": False,
        "sources": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--classification-workers", type=int, default=1)
    parser.add_argument("--maximum-source-restarts", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-sources", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "name": config.name,
                    "output_dir": str(config.output_dir),
                    "sources": len(config.sources),
                    "target_text_bytes": sum(
                        source.target_text_bytes for source in config.sources
                    ),
                    "language_targets": dict(
                        Counter(
                            {
                                language: sum(
                                    source.target_text_bytes
                                    for source in config.sources
                                    if source.language == language
                                )
                                for language in ALLOWED_LANGUAGES
                            }
                        )
                    ),
                    "text_conversion": "none",
                    "synthetic_training_content": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.probe_sources:
        print(
            json.dumps(
                probe_sources(args.config),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    manifest = build_with_retries(
        args.config,
        project_root=args.project_root,
        classification_workers=args.classification_workers,
        maximum_source_restarts=args.maximum_source_restarts,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
