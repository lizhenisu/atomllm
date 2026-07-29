"""Build resumable per-language public pretraining token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from atomllm.data.acquisition import chinese_script_classifier_identity
from atomllm.data.public_pretraining_plan import DEFAULT_PLAN, load_plan
from atomllm.data.public_tokenizer_corpus import (
    Source,
    _accepted_text,
    _accepted_text_worker,
    _dataset_state_cursor,
    _is_transient_source_error,
    _initialize_acceptance_worker,
    _iter_source,
    _resume_source_dataset,
    _reset_huggingface_http_client,
    _usable_iterator_checkpoint,
    load_config as load_source_registry,
)
from atomllm.tokenizer.evaluation import (
    TokenizerEvaluationError,
    verify_tokenizer_directory,
)


FORMAT_VERSION = "public-document-bos-eos-group-v1"
GROUPS = ("en", "code", "zh-Hans")
STATE_NAME = "state.json"
MANIFEST_NAME = "manifest.json"
COMPLETED_NAME = "COMPLETED"


class PublicTokenShardError(RuntimeError):
    """Raised when public token shard construction is unsafe or incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_priority(source: Source) -> tuple[int, str]:
    priorities = {
        "encyclopedia": 0,
        "science": 1,
        "math": 2,
        "code": 3,
        "general": 4,
    }
    return priorities[source.content_type], source.source_id


def _padded_length(token_count: int, sequence_length: int) -> int:
    if token_count <= 0 or sequence_length <= 0:
        raise ValueError("token_count and sequence_length must be positive")
    return ((token_count + sequence_length - 1) // sequence_length) * sequence_length


def _verify_shard(directory: Path, item: Mapping[str, Any]) -> None:
    documents = item.get("document_count")
    tokens = item.get("token_count")
    content_tokens = item.get("content_token_count")
    padding = item.get("padding_tokens")
    if any(
        type(value) is not int or value < 0
        for value in (documents, tokens, content_tokens, padding)
    ):
        raise PublicTokenShardError("shard counters are invalid")
    if documents <= 0 or tokens <= 0 or content_tokens <= 0:
        raise PublicTokenShardError("completed shard must be non-empty")
    if tokens != content_tokens + padding:
        raise PublicTokenShardError("shard padding accounting is inconsistent")
    for key, bytes_per_item in (
        ("token_file", 2),
        ("index_file", 16),
        ("digest_file", 32),
    ):
        metadata = item.get(key)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
            raise PublicTokenShardError(f"shard {key} metadata is invalid")
        path = directory / metadata["name"]
        if not path.is_file() or path.stat().st_size != metadata.get("size_bytes"):
            raise PublicTokenShardError(f"shard {key} is missing or truncated")
        if _sha256(path) != metadata.get("sha256"):
            raise PublicTokenShardError(f"shard {key} hash mismatch")
        expected_items = tokens if key == "token_file" else documents
        if path.stat().st_size != expected_items * bytes_per_item:
            raise PublicTokenShardError(f"shard {key} size accounting is invalid")


@dataclass
class _ShardWriter:
    output_dir: Path
    shard_index: int
    sequence_length: int
    eos_token_id: int
    source_id: str

    def __post_init__(self) -> None:
        base = f"part-{self.shard_index:05d}"
        self.token_final = self.output_dir / f"{base}.bin"
        self.index_final = self.output_dir / f"{base}.idx"
        self.digest_final = self.output_dir / f"{base}.digests"
        self.token_tmp = self.output_dir / f".{base}.bin.tmp"
        self.index_tmp = self.output_dir / f".{base}.idx.tmp"
        self.digest_tmp = self.output_dir / f".{base}.digests.tmp"
        for path in (self.token_tmp, self.index_tmp, self.digest_tmp):
            path.unlink(missing_ok=True)
        self.token_handle = self.token_tmp.open("wb")
        self.index_handle = self.index_tmp.open("wb")
        self.digest_handle = self.digest_tmp.open("wb")
        self.document_count = 0
        self.content_token_count = 0

    def append(self, ids: list[int], digest: bytes) -> None:
        document_tokens = np.empty(len(ids) + 2, dtype="<u2")
        document_tokens[0] = 2
        document_tokens[-1] = self.eos_token_id
        document_tokens[1:-1] = ids
        document_tokens.tofile(self.token_handle)
        np.asarray(
            (self.content_token_count, len(document_tokens)), dtype="<u8"
        ).tofile(self.index_handle)
        self.digest_handle.write(digest)
        self.document_count += 1
        self.content_token_count += len(document_tokens)

    def finish(self) -> dict[str, Any]:
        if self.document_count <= 0:
            raise PublicTokenShardError("cannot finish an empty token shard")
        padded = _padded_length(self.content_token_count, self.sequence_length)
        padding = padded - self.content_token_count
        if padding:
            np.full(padding, self.eos_token_id, dtype="<u2").tofile(self.token_handle)
        for handle in (self.token_handle, self.index_handle, self.digest_handle):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        os.replace(self.token_tmp, self.token_final)
        os.replace(self.index_tmp, self.index_final)
        os.replace(self.digest_tmp, self.digest_final)

        def metadata(path: Path, dtype: str) -> dict[str, Any]:
            return {
                "name": path.name,
                "dtype": dtype,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

        return {
            "shard_index": self.shard_index,
            "source_id": self.source_id,
            "document_count": self.document_count,
            "content_token_count": self.content_token_count,
            "padding_tokens": padding,
            "token_count": padded,
            "token_file": metadata(self.token_final, "uint16-le"),
            "index_file": metadata(self.index_final, "uint64-le[document,2]"),
            "digest_file": metadata(self.digest_final, "sha256-bytes"),
        }

    def cleanup(self) -> None:
        for name in ("token_handle", "index_handle", "digest_handle"):
            handle = getattr(self, name, None)
            if handle is not None and not handle.closed:
                handle.close()
        for path in (self.token_tmp, self.index_tmp, self.digest_tmp):
            path.unlink(missing_ok=True)


@dataclass
class _BuildRetryCache:
    """Process-local proof and dedup state reused across transient retries."""

    output_dir: Path | None = None
    identity: dict[str, Any] | None = None
    seen_digests: set[bytes] | None = None


def _load_digests(output_dir: Path, shards: Iterable[Mapping[str, Any]]) -> set[bytes]:
    values: set[bytes] = set()
    for shard in shards:
        path = output_dir / shard["digest_file"]["name"]
        raw = path.read_bytes()
        if len(raw) % 32:
            raise PublicTokenShardError("digest shard is truncated")
        values.update(raw[index : index + 32] for index in range(0, len(raw), 32))
    return values


def _identity(
    *,
    plan_sha256: str,
    tokenizer_manifest_sha256: str,
    tokenizer_sha256: str,
    group: str,
    workers: int,
    classifier_identity: Mapping[str, str],
    training_split: str,
    validation_status: str,
    gpu_selection_report_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "plan_sha256": plan_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "group": group,
        "workers": workers,
        "chinese_script_classifier": dict(classifier_identity),
        "validation_exclusion": None,
        "training_split": training_split,
        "validation_status": validation_status,
        "gpu_selection_report_sha256": gpu_selection_report_sha256,
        "token_dtype": "uint16-le",
        "document_boundaries": ["<bos>", "<eos>"],
        "synthetic_training_content": False,
        "local_text_conversion": "none",
        "local_privacy_filtering": "none",
    }


def build_group(
    *,
    plan_path: Path,
    tokenizer_dir: Path,
    output_root: Path,
    group: str,
    workers: int = 32,
    encode_batch_size: int = 256,
    gpu_selection_report_sha256: str | None = None,
    project_root: Path = Path("."),
    _retry_cache: _BuildRetryCache | None = None,
) -> dict[str, Any]:
    if group not in GROUPS:
        raise PublicTokenShardError(f"group must be one of {GROUPS}")
    cpu_count = os.cpu_count() or 1
    if type(workers) is not int or not 1 <= workers <= cpu_count:
        raise PublicTokenShardError(f"workers must be in [1, {cpu_count}]")
    if type(encode_batch_size) is not int or encode_batch_size <= 0:
        raise PublicTokenShardError("encode_batch_size must be positive")
    root = project_root.resolve()
    resolved_plan = (root / plan_path).resolve()
    resolved_tokenizer = (root / tokenizer_dir).resolve()
    output_dir = (root / output_root / group).resolve()
    if not all(
        path.is_relative_to(root)
        for path in (resolved_plan, resolved_tokenizer, output_dir)
    ):
        raise PublicTokenShardError("public token shard paths must remain in project")
    plan = load_plan(resolved_plan, project_root=root)
    registry = load_source_registry(root / plan.source_registry)
    registered_sources = (
        *registry.sources,
        *getattr(plan, "supplemental_sources", ()),
    )
    sources = sorted(
        (
            replace(
                source,
                **getattr(plan, "source_overrides", {}).get(source.source_id, {}),
            )
            for source in registered_sources
            if source.language == group
        ),
        key=_source_priority,
    )
    if not sources:
        raise PublicTokenShardError(f"source registry has no sources for {group}")
    # Rayon reads this environment when the tokenizer's worker pool is first
    # initialized. Set it before loading tokenizer.json so this function owns
    # its CPU budget even if tokenizers changes initialization behavior.
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        tokenizer, tokenizer_manifest, tokenizer_manifest_path = (
            verify_tokenizer_directory(resolved_tokenizer)
        )
    except TokenizerEvaluationError as error:
        raise PublicTokenShardError(str(error)) from error
    if tokenizer_manifest.get("training_eligible") is not True:
        raise PublicTokenShardError("tokenizer is not training eligible")
    tokenizer_path = resolved_tokenizer / "tokenizer.json"
    tokenizer_sha = _sha256(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise PublicTokenShardError("tokenizer vocabulary does not fit uint16")
    for token, expected in (("<unk>", 1), ("<bos>", 2), ("<eos>", 3)):
        if tokenizer.token_to_id(token) != expected:
            raise PublicTokenShardError(f"tokenizer {token} ID mismatch")
    tokenizer.encode_special_tokens = True
    identity = _identity(
        plan_sha256=_sha256(resolved_plan),
        tokenizer_manifest_sha256=_sha256(tokenizer_manifest_path),
        tokenizer_sha256=tokenizer_sha,
        group=group,
        workers=workers,
        classifier_identity=chinese_script_classifier_identity(),
        training_split=plan.training_split,
        validation_status=plan.validation_status,
        gpu_selection_report_sha256=gpu_selection_report_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    completed_path = output_dir / COMPLETED_NAME
    if manifest_path.exists() or completed_path.exists():
        return verify_group(output_dir, expected_identity=identity)
    state_path = output_dir / STATE_NAME
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise PublicTokenShardError("resume state identity mismatch")
    else:
        state = {
            "schema_version": 1,
            "identity": identity,
            "source_index": 0,
            "source_order": [source.source_id for source in sources],
            "source_records_seen": {source.source_id: 0 for source in sources},
            "source_iterator_states": {},
            "source_content_tokens": {source.source_id: 0 for source in sources},
            "source_documents": {source.source_id: 0 for source in sources},
            "source_effective_target_tokens": {},
            "source_exhausted": {},
            "carried_shortfall_tokens": 0,
            "duplicate_documents": 0,
            "rejected_documents": 0,
            "shards": [],
        }
        _write_json(state_path, state)
    state.setdefault("source_iterator_states", {})
    state.setdefault("source_order", [source.source_id for source in sources])
    state.setdefault("source_effective_target_tokens", {})
    state.setdefault("source_exhausted", {})
    state.setdefault("carried_shortfall_tokens", 0)
    if state["source_order"] != [source.source_id for source in sources]:
        raise PublicTokenShardError("resume state source order mismatch")
    shards = state.get("shards")
    if not isinstance(shards, list):
        raise PublicTokenShardError("resume state shards are invalid")
    expected_digest_count = sum(shard.get("document_count", -1) for shard in shards)
    cached_digests = None if _retry_cache is None else _retry_cache.seen_digests
    retry_cache_valid = (
        _retry_cache is not None
        and _retry_cache.output_dir == output_dir
        and _retry_cache.identity == identity
        and cached_digests is not None
        and len(cached_digests) == expected_digest_count
    )
    if retry_cache_valid:
        seen_digests = cached_digests
        print(
            "[public-token-shards] reuse process-local verified shards "
            f"group={group} shards={len(shards)} digests={len(seen_digests)}",
            flush=True,
        )
    else:
        for shard in shards:
            _verify_shard(output_dir, shard)
        seen_digests = _load_digests(output_dir, shards)
        if len(seen_digests) != expected_digest_count:
            raise PublicTokenShardError("committed shard digests are not unique")
        if _retry_cache is not None:
            _retry_cache.output_dir = output_dir
            _retry_cache.identity = identity
            _retry_cache.seen_digests = seen_digests
    shard_capacity = plan.shard_token_capacity
    sequence_length = plan.sequence_length
    writer: _ShardWriter | None = None
    writer_digests: set[bytes] = set()
    try:
        for source_index in range(state["source_index"], len(sources)):
            source = sources[source_index]
            requested_target = plan.source_target_tokens[source.source_id]
            target = requested_target + state["carried_shortfall_tokens"]
            prior_effective_target = state["source_effective_target_tokens"].get(
                source.source_id
            )
            if prior_effective_target is not None and prior_effective_target != target:
                raise PublicTokenShardError("resume state effective target mismatch")
            state["source_effective_target_tokens"][source.source_id] = target
            source_tokens = state["source_content_tokens"][source.source_id]
            source_documents = state["source_documents"][source.source_id]
            seen = state["source_records_seen"][source.source_id]
            if source_tokens >= target:
                state["source_exhausted"][source.source_id] = False
                state["carried_shortfall_tokens"] = 0
                state["source_index"] = source_index + 1
                _write_json(state_path, state)
                continue
            print(
                f"[public-token-shards] group={group} source={source.source_id} "
                f"resume_seen={seen} resume_tokens={source_tokens} target={target}",
                flush=True,
            )
            raw_iterator_checkpoint = state["source_iterator_states"].get(
                source.source_id
            )
            iterator_checkpoint = _usable_iterator_checkpoint(
                raw_iterator_checkpoint, seen
            )
            if iterator_checkpoint is None:
                if raw_iterator_checkpoint is not None:
                    print(
                        "[public-token-shards] invalid iterator checkpoint; "
                        f"safe_rebase_source={source.source_id} records_seen={seen}",
                        flush=True,
                    )
                state["source_iterator_states"].pop(source.source_id, None)
            base_skip_records = (
                seen
                if iterator_checkpoint is None
                else iterator_checkpoint["base_skip_records"]
            )
            replay_through_records_seen = (
                seen
                if iterator_checkpoint is None
                else iterator_checkpoint.get("replay_through_records_seen", seen)
            )
            saved_dataset_state = (
                None
                if iterator_checkpoint is None
                else iterator_checkpoint["dataset_state"]
            )
            saved_dataset_state_records_seen = (
                seen
                if iterator_checkpoint is None
                else iterator_checkpoint.get(
                    "dataset_state_records_seen", iterator_checkpoint["records_seen"]
                )
            )
            saved_dataset_state_cursor = _dataset_state_cursor(saved_dataset_state)
            source_dataset = (
                _iter_source(source, seen)
                if iterator_checkpoint is None
                else _resume_source_dataset(
                    source,
                    records_seen=seen,
                    iterator_checkpoint=iterator_checkpoint,
                )
            )
            can_checkpoint_iterator = callable(
                getattr(source_dataset, "state_dict", None)
            )
            source_iterator = iter(source_dataset)
            iterator = enumerate(
                source_iterator, start=saved_dataset_state_records_seen + 1
            )
            acceptance_executor = (
                ProcessPoolExecutor(
                    max_workers=min(workers, 32),
                    mp_context=get_context("spawn"),
                    initializer=_initialize_acceptance_worker,
                    initargs=(source,),
                )
                if source.language == "zh-Hans" and workers > 1
                else None
            )
            try:
                while source_tokens < target:
                    batch_start_seen = seen
                    batch_start_dataset_state = (
                        source_dataset.state_dict() if can_checkpoint_iterator else None
                    )
                    batch_start_dataset_cursor = _dataset_state_cursor(
                        batch_start_dataset_state
                    )
                    records: list[tuple[int, Mapping[str, Any]]] = []
                    for _ in range(encode_batch_size):
                        try:
                            records.append(next(iterator))
                        except StopIteration:
                            break
                    if not records:
                        break
                    accepted_records = (
                        acceptance_executor.map(
                            _accepted_text_worker,
                            (record for _, record in records),
                            chunksize=8,
                        )
                        if acceptance_executor is not None
                        else (_accepted_text(source, record) for _, record in records)
                    )
                    pending: list[tuple[str, bytes]] = []
                    pending_digests: set[bytes] = set()
                    events: list[tuple[str, int, int | None]] = []
                    for (position, _record), accepted in zip(
                        records, accepted_records, strict=True
                    ):
                        is_resume_replay = position <= replay_through_records_seen
                        if accepted is None:
                            events.append(("rejected", position, None))
                            continue
                        text, _ = accepted
                        digest = hashlib.sha256(text.encode("utf-8")).digest()
                        if digest in seen_digests or digest in pending_digests:
                            events.append(("duplicate", position, None))
                            continue
                        if is_resume_replay:
                            raise PublicTokenShardError(
                                f"resume replay digest is missing for {source.source_id}"
                            )
                        pending_index = len(pending)
                        pending.append((text, digest))
                        pending_digests.add(digest)
                        events.append(("accepted", position, pending_index))
                    encodings = tokenizer.encode_batch(
                        [text for text, _ in pending], add_special_tokens=False
                    )
                    target_reached = False
                    for event, position, pending_index in events:
                        is_resume_replay = position <= replay_through_records_seen
                        if event == "rejected":
                            if not is_resume_replay:
                                state["rejected_documents"] += 1
                            seen = position
                            continue
                        if event == "duplicate":
                            if not is_resume_replay:
                                state["duplicate_documents"] += 1
                            seen = position
                            continue
                        if pending_index is None:
                            raise AssertionError(
                                "accepted event is missing encoding index"
                            )
                        _text, digest = pending[pending_index]
                        encoding = encodings[pending_index]
                        if encoding.ids.count(1):
                            raise PublicTokenShardError(
                                f"tokenizer emitted <unk> for {source.source_id}"
                            )
                        document_tokens = len(encoding.ids) + 2
                        if writer is not None and (
                            writer.content_token_count + document_tokens
                            > shard_capacity
                        ):
                            item = writer.finish()
                            shards.append(item)
                            if (
                                batch_start_dataset_state is not None
                                and batch_start_dataset_cursor
                                > saved_dataset_state_cursor
                            ):
                                saved_dataset_state = batch_start_dataset_state
                                saved_dataset_state_records_seen = batch_start_seen
                                saved_dataset_state_cursor = batch_start_dataset_cursor
                            state["source_records_seen"][source.source_id] = (
                                batch_start_seen
                                if saved_dataset_state is not None
                                else seen
                            )
                            state["source_content_tokens"][source.source_id] = (
                                source_tokens
                            )
                            state["source_documents"][source.source_id] = (
                                source_documents
                            )
                            if saved_dataset_state is not None:
                                state["source_iterator_states"][source.source_id] = {
                                    "records_seen": batch_start_seen,
                                    "base_skip_records": base_skip_records,
                                    "dataset_state_records_seen": (
                                        saved_dataset_state_records_seen
                                    ),
                                    "replay_through_records_seen": seen,
                                    "dataset_state": saved_dataset_state,
                                }
                            _write_json(state_path, state)
                            writer_digests.clear()
                            print(
                                f"[public-token-shards] "
                                f"finish shard={item['shard_index']} "
                                f"group={group} tokens={item['token_count']}",
                                flush=True,
                            )
                            writer = None
                        if writer is None:
                            writer = _ShardWriter(
                                output_dir=output_dir,
                                shard_index=len(shards),
                                sequence_length=sequence_length,
                                eos_token_id=3,
                                source_id=source.source_id,
                            )
                        writer.append(encoding.ids, digest)
                        seen_digests.add(digest)
                        writer_digests.add(digest)
                        source_tokens += document_tokens
                        source_documents += 1
                        seen = position
                        if source_tokens >= target:
                            target_reached = True
                            break
                    if target_reached:
                        break
            finally:
                if acceptance_executor is not None:
                    acceptance_executor.shutdown(cancel_futures=True)
                close_iterator = getattr(source_iterator, "close", None)
                if callable(close_iterator):
                    close_iterator()
            if writer is not None:
                item = writer.finish()
                shards.append(item)
                writer = None
                print(
                    f"[public-token-shards] finish source shard={item['shard_index']} "
                    f"source={source.source_id} tokens={item['token_count']}",
                    flush=True,
                )
            state["source_records_seen"][source.source_id] = seen
            state["source_content_tokens"][source.source_id] = source_tokens
            state["source_documents"][source.source_id] = source_documents
            exhausted = source_tokens < target
            state["source_exhausted"][source.source_id] = exhausted
            state["carried_shortfall_tokens"] = (
                target - source_tokens if exhausted else 0
            )
            state["source_index"] = source_index + 1
            _write_json(state_path, state)
            writer_digests.clear()
            if exhausted:
                print(
                    "[public-token-shards] source exhausted; "
                    f"source={source.source_id} actual_tokens={source_tokens} "
                    f"effective_target={target} "
                    f"carry_tokens={state['carried_shortfall_tokens']}",
                    flush=True,
                )
        if state["carried_shortfall_tokens"]:
            raise PublicTokenShardError(
                f"group exhausted before language target: {group} "
                f"shortfall={state['carried_shortfall_tokens']}"
            )
        manifest = {
            "schema_version": 1,
            "format_version": FORMAT_VERSION,
            "dataset_id": (
                f"public-token-group-{group}-"
                f"{hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:12]}"
            ),
            "identity": identity,
            "group": group,
            "sequence_length": sequence_length,
            "token_dtype": "uint16-le",
            "vocab_size": vocab_size,
            "document_count": sum(item["document_count"] for item in shards),
            "content_token_count": sum(item["content_token_count"] for item in shards),
            "padding_token_count": sum(item["padding_tokens"] for item in shards),
            "token_count": sum(item["token_count"] for item in shards),
            "source_target_tokens": {
                source.source_id: plan.source_target_tokens[source.source_id]
                for source in sources
            },
            "source_order": state["source_order"],
            "source_effective_target_tokens": state["source_effective_target_tokens"],
            "source_exhausted": state["source_exhausted"],
            "source_content_tokens": state["source_content_tokens"],
            "source_documents": state["source_documents"],
            "duplicate_documents": state["duplicate_documents"],
            "rejected_documents": state["rejected_documents"],
            "training_split": plan.training_split,
            "validation_status": plan.validation_status,
            "shards": shards,
            "training_eligible": True,
            "synthetic_training_content": False,
            "local_text_conversion": "none",
            "local_privacy_filtering": "none",
        }
        _write_json(manifest_path, manifest)
        completed_path.write_text(
            f"{_sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
        )
        state_path.unlink()
        return manifest
    except BaseException:
        seen_digests.difference_update(writer_digests)
        if writer is not None:
            writer.cleanup()
        raise


def build_group_with_retries(
    *,
    maximum_source_restarts: int = 1000,
    **kwargs: Any,
) -> dict[str, Any]:
    if type(maximum_source_restarts) is not int or maximum_source_restarts < 0:
        raise PublicTokenShardError(
            "maximum_source_restarts must be a non-negative integer"
        )
    restarts = 0
    retry_cache = _BuildRetryCache()
    while True:
        try:
            return build_group(**kwargs, _retry_cache=retry_cache)
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
                "[public-token-shards] transient source failure; "
                f"restart={restarts}/{maximum_source_restarts} "
                f"delay_seconds={delay} error={type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(delay)


def verify_group(
    directory: Path, *, expected_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest_path = directory / MANIFEST_NAME
    completed_path = directory / COMPLETED_NAME
    if not manifest_path.is_file() or not completed_path.is_file():
        raise PublicTokenShardError("public token group is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise PublicTokenShardError("public token group format is invalid")
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        raise PublicTokenShardError("public token group identity mismatch")
    if completed_path.read_text(encoding="utf-8") != (
        f"{_sha256(manifest_path)}  {MANIFEST_NAME}\n"
    ):
        raise PublicTokenShardError("public token group marker is invalid")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PublicTokenShardError("public token group has no shards")
    for item in shards:
        _verify_shard(directory, item)
    if manifest.get("token_count") != sum(item["token_count"] for item in shards):
        raise PublicTokenShardError("public token group total is invalid")
    return manifest


def tokenizer_from_gpu_selection(
    selection_dir: Path,
    *,
    project_root: Path = Path("."),
) -> tuple[Path, str]:
    """Resolve a training-eligible tokenizer from completed quality/GPU evidence."""
    root = project_root.resolve()
    directory = (root / selection_dir).resolve()
    if not directory.is_relative_to(root):
        raise PublicTokenShardError("tokenizer selection path escapes project root")
    report_path = directory / "report.json"
    completed_path = directory / COMPLETED_NAME
    if not report_path.is_file() or not completed_path.is_file():
        raise PublicTokenShardError("GPU tokenizer selection is incomplete")
    report_sha = _sha256(report_path)
    if completed_path.read_text(encoding="utf-8") != f"{report_sha}  report.json\n":
        raise PublicTokenShardError("GPU tokenizer selection marker is invalid")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("training_eligible") is not True
        or report.get("gpu_confirmed") is not True
    ):
        raise PublicTokenShardError("tokenizer did not pass quality and GPU gates")
    relative = report.get("selected_tokenizer_dir")
    if not isinstance(relative, str):
        raise PublicTokenShardError("selected tokenizer path is invalid")
    tokenizer_dir = (root / relative).resolve()
    if not tokenizer_dir.is_relative_to(root):
        raise PublicTokenShardError("selected tokenizer path escapes project root")
    if _sha256(tokenizer_dir / "tokenizer.json") != report.get(
        "selected_tokenizer_sha256"
    ):
        raise PublicTokenShardError("selected tokenizer hash mismatch")
    if _sha256(tokenizer_dir / MANIFEST_NAME) != report.get(
        "selected_tokenizer_manifest_sha256"
    ):
        raise PublicTokenShardError("selected tokenizer manifest hash mismatch")
    return tokenizer_dir, report_sha


def assemble_groups(
    *,
    group_root: Path,
    output_dir: Path,
    plan_path: Path = DEFAULT_PLAN,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    root = project_root.resolve()
    groups_dir = (root / group_root).resolve()
    final_dir = (root / output_dir).resolve()
    resolved_plan = (root / plan_path).resolve()
    if not all(
        path.is_relative_to(root) for path in (groups_dir, final_dir, resolved_plan)
    ):
        raise PublicTokenShardError("assembly paths must remain inside project root")
    plan = load_plan(resolved_plan, project_root=root)
    manifests = {group: verify_group(groups_dir / group) for group in GROUPS}
    shared_identity_fields = (
        "plan_sha256",
        "tokenizer_manifest_sha256",
        "tokenizer_sha256",
        "gpu_selection_report_sha256",
        "validation_exclusion",
        "training_split",
        "validation_status",
        "token_dtype",
        "document_boundaries",
        "synthetic_training_content",
        "local_text_conversion",
        "local_privacy_filtering",
    )
    baseline = manifests[GROUPS[0]]["identity"]
    if baseline.get("plan_sha256") != _sha256(resolved_plan):
        raise PublicTokenShardError("group plan does not match assembly plan")
    selection_sha = baseline.get("gpu_selection_report_sha256")
    if (
        not isinstance(selection_sha, str)
        or len(selection_sha) != 64
        or any(character not in "0123456789abcdef" for character in selection_sha)
    ):
        raise PublicTokenShardError("groups lack a verified GPU tokenizer selection")
    for group, manifest in manifests.items():
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise PublicTokenShardError(f"group identity is invalid: {group}")
        for field in shared_identity_fields:
            if identity.get(field) != baseline.get(field):
                raise PublicTokenShardError(
                    f"group identity mismatch for {field}: {group}"
                )
    if len({manifest.get("vocab_size") for manifest in manifests.values()}) != 1:
        raise PublicTokenShardError("group tokenizer vocabulary sizes differ")
    for group, manifest in manifests.items():
        source_targets = manifest.get("source_target_tokens")
        source_tokens = manifest.get("source_content_tokens")
        expected_targets = (
            {
                source_id: target
                for source_id, target in plan.source_target_tokens.items()
                if source_id in source_targets
            }
            if isinstance(source_targets, dict)
            else {}
        )
        if source_targets != expected_targets or not expected_targets:
            raise PublicTokenShardError(f"group source targets mismatch: {group}")
        if not isinstance(source_tokens, dict) or set(source_tokens) != set(
            source_targets
        ):
            raise PublicTokenShardError(
                f"group source token accounting mismatch: {group}"
            )
        source_order = manifest.get("source_order")
        effective_targets = manifest.get("source_effective_target_tokens")
        exhausted_sources = manifest.get("source_exhausted")
        if (
            isinstance(source_order, list)
            and isinstance(effective_targets, dict)
            and isinstance(exhausted_sources, dict)
        ):
            if (
                len(source_order) != len(set(source_order))
                or set(source_order) != set(source_targets)
                or set(effective_targets) != set(source_targets)
                or set(exhausted_sources) != set(source_targets)
            ):
                raise PublicTokenShardError(
                    f"group source allocation schema mismatch: {group}"
                )
            carry = 0
            for source_id in source_order:
                requested = source_targets[source_id]
                actual = source_tokens[source_id]
                effective = effective_targets[source_id]
                exhausted = exhausted_sources[source_id]
                if (
                    type(requested) is not int
                    or type(actual) is not int
                    or type(effective) is not int
                    or type(exhausted) is not bool
                    or effective != requested + carry
                ):
                    raise PublicTokenShardError(
                        f"group source allocation is invalid: {group}"
                    )
                if exhausted:
                    if actual >= effective:
                        raise PublicTokenShardError(
                            f"group source exhaustion is invalid: {group}"
                        )
                    carry = effective - actual
                else:
                    if actual < effective:
                        raise PublicTokenShardError(
                            f"group source target was not reached: {group}"
                        )
                    carry = 0
            if carry:
                raise PublicTokenShardError(
                    f"group language target was not reached: {group}"
                )
        elif any(
            source_tokens[source_id] < target
            for source_id, target in source_targets.items()
        ):
            raise PublicTokenShardError(f"group source target was not reached: {group}")
        if sum(source_targets.values()) != plan.language_target_tokens[group]:
            raise PublicTokenShardError(f"group language target mismatch: {group}")
    if final_dir.exists():
        from atomllm.training.formal_token_shards import verify_formal_token_shards

        return verify_formal_token_shards(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_dir.parent / f".{final_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise PublicTokenShardError("assembly temporary directory already exists")
    temporary.mkdir()
    final_shards = []
    try:
        output_index = 0
        for group in GROUPS:
            group_dir = groups_dir / group
            for item in manifests[group]["shards"]:
                base = f"part-{output_index:05d}"
                token_name = f"{base}.bin"
                index_name = f"{base}.idx"
                source_token = group_dir / item["token_file"]["name"]
                source_index = group_dir / item["index_file"]["name"]
                os.link(source_token, temporary / token_name)
                os.link(source_index, temporary / index_name)
                token_metadata = {
                    "name": token_name,
                    "dtype": "uint16-le",
                    "size_bytes": item["token_file"]["size_bytes"],
                    "sha256": item["token_file"]["sha256"],
                }
                index_metadata = {
                    "name": index_name,
                    "dtype": "uint64-le",
                    "shape": [item["document_count"], 2],
                    "size_bytes": item["index_file"]["size_bytes"],
                    "sha256": item["index_file"]["sha256"],
                }
                final_shards.append(
                    {
                        "source_name": (f"{group}/{item['token_file']['name']}"),
                        "source_sha256": item["token_file"]["sha256"],
                        "source_id": item["source_id"],
                        "language_group": group,
                        "document_count": item["document_count"],
                        "content_token_count": item["content_token_count"],
                        "padding_tokens": item["padding_tokens"],
                        "token_count": item["token_count"],
                        "token_file": token_metadata,
                        "index_file": index_metadata,
                    }
                )
                output_index += 1
        group_manifest_sha256 = {
            group: _sha256(groups_dir / group / MANIFEST_NAME) for group in GROUPS
        }
        identity = {
            "format_version": "document-bos-eos-sharded-v2",
            "public_group_format_version": FORMAT_VERSION,
            "plan_sha256": baseline["plan_sha256"],
            "tokenizer_manifest_sha256": baseline["tokenizer_manifest_sha256"],
            "tokenizer_sha256": baseline["tokenizer_sha256"],
            "gpu_selection_report_sha256": baseline.get("gpu_selection_report_sha256"),
            "validation_exclusion": baseline.get("validation_exclusion"),
            "training_split": baseline["training_split"],
            "validation_status": baseline["validation_status"],
            "group_manifest_sha256": group_manifest_sha256,
            "token_dtype": "uint16-le",
            "encode_special_tokens_as_text": True,
            "synthetic_training_content": False,
            "local_text_conversion": "none",
            "local_privacy_filtering": "none",
        }
        identity_sha = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
        manifest = {
            "schema_version": 1,
            "format_version": "document-bos-eos-sharded-v2",
            "dataset_id": f"public-token-shards-100b-{identity_sha[:12]}",
            "identity_sha256": identity_sha,
            "identity": identity,
            "data_version": {
                "plan_sha256": baseline["plan_sha256"],
                "training_eligible": True,
            },
            "tokenizer": {
                "tokenizer_manifest_sha256": baseline["tokenizer_manifest_sha256"],
                "tokenizer_sha256": baseline["tokenizer_sha256"],
                "vocab_size": manifests[GROUPS[0]]["vocab_size"],
            },
            "token_dtype": "uint16-le",
            "split": "train",
            "sequence_length": manifests[GROUPS[0]]["sequence_length"],
            "encode_special_tokens_as_text": True,
            "index_columns": ["token_offset", "token_count"],
            "document_count": sum(
                manifest["document_count"] for manifest in manifests.values()
            ),
            "content_token_count": sum(
                manifest["content_token_count"] for manifest in manifests.values()
            ),
            "padding_token_count": sum(
                manifest["padding_token_count"] for manifest in manifests.values()
            ),
            "token_count": sum(
                manifest["token_count"] for manifest in manifests.values()
            ),
            "language_content_tokens": {
                group: manifests[group]["content_token_count"] for group in GROUPS
            },
            "source_content_tokens": {
                source_id: count
                for group in GROUPS
                for source_id, count in manifests[group][
                    "source_content_tokens"
                ].items()
            },
            "group_manifest_sha256": group_manifest_sha256,
            "validation_exclusion": baseline.get("validation_exclusion"),
            "training_split": baseline["training_split"],
            "validation_status": baseline["validation_status"],
            "shards": final_shards,
            "formal_training_eligible": True,
            "public_training_eligible": True,
            "synthetic_training_content": False,
            "local_text_conversion": "none",
            "local_privacy_filtering": "none",
        }
        manifest_path = temporary / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        (temporary / COMPLETED_NAME).write_text(
            f"{_sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
        )
        from atomllm.training.formal_token_shards import verify_formal_token_shards

        # Verify every hard-linked shard while the artifact is still private.
        # Publishing the directory first exposes COMPLETED to downstream stages
        # before this full verification finishes and causes duplicate scans.
        verified = verify_formal_token_shards(temporary)
        os.replace(temporary, final_dir)
        return verified
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--tokenizer-selection-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--assemble-output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--maximum-source-restarts", type=int, default=1000)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.assemble_output_dir is not None:
        if (
            args.group is not None
            or args.tokenizer_dir is not None
            or args.tokenizer_selection_dir is not None
        ):
            parser.error("assembly cannot be combined with tokenizer/group options")
        manifest = assemble_groups(
            group_root=args.output_root,
            output_dir=args.assemble_output_dir,
            plan_path=args.plan,
            project_root=args.project_root,
        )
        print(_canonical_json(manifest))
        return 0
    if args.group is None:
        parser.error("group building requires --group")
    if (args.tokenizer_dir is None) == (args.tokenizer_selection_dir is None):
        parser.error(
            "group building requires exactly one of --tokenizer-dir or "
            "--tokenizer-selection-dir"
        )
    selection_sha = None
    tokenizer_dir = args.tokenizer_dir
    if args.tokenizer_selection_dir is not None:
        tokenizer_dir, selection_sha = tokenizer_from_gpu_selection(
            args.tokenizer_selection_dir,
            project_root=args.project_root,
        )
    assert tokenizer_dir is not None
    manifest = build_group_with_retries(
        plan_path=args.plan,
        tokenizer_dir=tokenizer_dir,
        output_root=args.output_root,
        group=args.group,
        workers=args.workers,
        encode_batch_size=args.encode_batch_size,
        gpu_selection_report_sha256=selection_sha,
        project_root=args.project_root,
        maximum_source_restarts=args.maximum_source_restarts,
    )
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
