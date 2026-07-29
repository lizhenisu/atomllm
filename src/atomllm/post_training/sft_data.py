"""Build the frozen stage-8 SFT dataset with assistant-only labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow.parquet as parquet
import yaml
from tokenizers import Tokenizer


SCHEMA_VERSION = 1
INPUT_FILE = "input_ids.u32"
LABEL_FILE = "labels.i32"
IGNORE_INDEX = -100
ROLE_TOKENS = {"system": 4, "user": 5, "assistant": 6}
PAD_ID = 0
BOS_ID = 2
EOT_ID = 8


class SFTDataError(RuntimeError):
    """Raised when stage-8 data cannot satisfy its frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _near_duplicate_key(messages: list[dict[str, str]]) -> str:
    """Match formatting-only variants without claiming semantic deduplication."""
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message["role"].encode())
        normalized = unicodedata.normalize("NFKC", message["content"]).casefold()
        # Arithmetic operators carry semantics and must not collapse distinct
        # expressions such as ``4 + 1`` and ``4 / 1`` into one near-duplicate.
        normalized = re.sub(
            r"[^\w+\-*/×÷]+",
            "",
            normalized,
            flags=re.UNICODE,
        )
        digest.update(normalized.encode())
        digest.update(b"\0")
    return digest.hexdigest()


_EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()


def _conversation(
    source: str,
    messages: Iterable[tuple[str, Any]],
    *,
    final_assistant_only: bool = False,
) -> dict[str, Any] | None:
    normalized: list[dict[str, str]] = []
    for role, raw_content in messages:
        content = _normalize_text(raw_content)
        if role not in ROLE_TOKENS or not content:
            return None
        if normalized and normalized[-1]["role"] == role and role != "system":
            return None
        normalized.append({"role": role, "content": content})
    if len(normalized) < 2 or normalized[-1]["role"] != "assistant":
        return None
    if not any(message["role"] == "user" for message in normalized):
        return None
    canonical = _canonical(normalized)
    assistant_indices = [
        index
        for index, message in enumerate(normalized)
        if message["role"] == "assistant"
    ]
    return {
        "conversation_id": hashlib.sha256(canonical.encode()).hexdigest(),
        "source": source,
        "messages": normalized,
        "target_message_indices": (
            [assistant_indices[-1]] if final_assistant_only else assistant_indices
        ),
    }


def _ultrachat(root: Path, *, split: str = "train") -> Iterator[dict[str, Any]]:
    if split not in {"train", "test"}:
        raise ValueError("UltraChat split must be train or test")
    for path in sorted((root / "data").glob(f"{split}_sft-*.parquet")):
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(columns=["messages"], batch_size=2048):
            for row in batch.to_pylist():
                result = _conversation(
                    "ultrachat-200k",
                    ((item["role"], item["content"]) for item in row["messages"]),
                )
                if result is not None:
                    yield result


def _oasst_split(path: Path, source: str) -> Iterator[dict[str, Any]]:
    table = parquet.read_table(path)
    rows = table.select(
        [
            "message_id",
            "parent_id",
            "message_tree_id",
            "text",
            "role",
            "deleted",
            "review_result",
            "rank",
            "tree_state",
        ]
    ).to_pylist()
    by_id = {row["message_id"]: row for row in rows}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots: list[dict[str, Any]] = []
    for row in rows:
        parent = row["parent_id"]
        if parent in by_id:
            children[parent].append(row)
        elif parent is None:
            roots.append(row)

    def valid(row: dict[str, Any]) -> bool:
        return (
            row["deleted"] is not True
            and row["review_result"] is not False
            and row["tree_state"] == "ready_for_export"
            and row["role"] in {"prompter", "assistant"}
            and bool(_normalize_text(row["text"]))
        )

    for root in sorted(roots, key=lambda row: row["message_tree_id"]):
        if not valid(root) or root["role"] != "prompter":
            continue
        path_rows = [root]
        current = root
        while True:
            expected = "assistant" if current["role"] == "prompter" else "prompter"
            candidates = [
                row
                for row in children.get(current["message_id"], [])
                if valid(row) and row["role"] == expected
            ]
            if not candidates:
                break
            current = min(
                candidates,
                key=lambda row: (
                    row["rank"] is None,
                    row["rank"] if row["rank"] is not None else 2**31,
                    row["message_id"],
                ),
            )
            path_rows.append(current)
        while path_rows and path_rows[-1]["role"] != "assistant":
            path_rows.pop()
        result = _conversation(
            source,
            (
                ("user" if row["role"] == "prompter" else "assistant", row["text"])
                for row in path_rows
            ),
        )
        if result is not None:
            yield result


def _oasst_all_assistant_paths(path: Path, source: str) -> Iterator[dict[str, Any]]:
    """Yield every valid assistant reply once with its complete ancestry."""
    table = parquet.read_table(path)
    rows = table.select(
        [
            "message_id",
            "parent_id",
            "message_tree_id",
            "text",
            "role",
            "deleted",
            "review_result",
            "tree_state",
        ]
    ).to_pylist()
    by_id = {row["message_id"]: row for row in rows}

    def valid(row: dict[str, Any]) -> bool:
        return (
            row["deleted"] is not True
            and row["review_result"] is not False
            and row["tree_state"] == "ready_for_export"
            and row["role"] in {"prompter", "assistant"}
            and bool(_normalize_text(row["text"]))
        )

    assistants = sorted(
        (row for row in rows if row["role"] == "assistant" and valid(row)),
        key=lambda row: row["message_id"],
    )
    for assistant in assistants:
        ancestry: list[dict[str, Any]] = []
        current: dict[str, Any] | None = assistant
        visited: set[str] = set()
        while current is not None:
            message_id = current["message_id"]
            if message_id in visited or not valid(current):
                ancestry = []
                break
            visited.add(message_id)
            ancestry.append(current)
            parent_id = current["parent_id"]
            current = by_id.get(parent_id) if parent_id is not None else None
        ancestry.reverse()
        if not ancestry or ancestry[0]["role"] != "prompter":
            continue
        result = _conversation(
            source,
            (
                ("user" if row["role"] == "prompter" else "assistant", row["text"])
                for row in ancestry
            ),
            final_assistant_only=True,
        )
        if result is not None:
            yield result


def _json_values(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict):
            yield value
        else:
            raise SFTDataError(f"unsupported JSON value in {path}")


def _coig_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    if {"instruction", "output"} <= row.keys():
        instruction = row.get("trans_instruction") or row["instruction"]
        input_text = row.get("trans_input") or row.get("input") or ""
        output = row.get("trans_output") or row["output"]
        prompt = (
            f"{instruction}\n\n{input_text}"
            if _normalize_text(input_text)
            else instruction
        )
        return prompt, output
    if {"textbox_q_instruction", "textbox_question", "textbox_answer"} <= row.keys():
        pieces = [
            row["textbox_q_instruction"],
            row.get("textbox_q_context", ""),
            row["textbox_question"],
        ]
        answer = str(row["textbox_answer"])
        analysis = row.get("textbox_answer_analysis")
        if analysis:
            answer = f"{answer}\n\n{analysis if isinstance(analysis, str) else json.dumps(analysis, ensure_ascii=False)}"
        return "\n\n".join(
            _normalize_text(piece) for piece in pieces if _normalize_text(piece)
        ), answer
    return None


def _coig(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(root.iterdir()):
        if path.suffix not in {".json", ".jsonl"}:
            continue
        for row in _json_values(path):
            pair = _coig_pair(row)
            if pair is None:
                continue
            result = _conversation("coig", (("user", pair[0]), ("assistant", pair[1])))
            if result is not None:
                yield result
    archive = root / "counterfactural_correction_multi_round_chat.tar.gz"
    # Stream compressed members in archive order. Name-sorted random access
    # repeatedly decompresses the gzip prefix and becomes quadratic. The final
    # conversations are independently sorted by their content SHA.
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            row = json.load(extracted)
            messages: list[tuple[str, str]] = []
            for key in sorted(
                (key for key in row if key.startswith("round_")),
                key=lambda key: int(key.split("_")[1]),
            ):
                response = row[key].get("response")
                try:
                    parsed = json.loads(response)
                except TypeError, json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("Q") and parsed.get("A"):
                    messages.extend((("user", parsed["Q"]), ("assistant", parsed["A"])))
            result = _conversation("coig", messages)
            if result is not None:
                yield result


def _encode(
    tokenizer: Tokenizer, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    tokens = [BOS_ID]
    labels = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = tokenizer.encode(message["content"], add_special_tokens=False).ids
        tokens.append(ROLE_TOKENS[role])
        labels.append(IGNORE_INDEX)
        tokens.extend(content)
        labels.extend(content if role == "assistant" else [IGNORE_INDEX] * len(content))
        tokens.append(EOT_ID)
        labels.append(EOT_ID if role == "assistant" else IGNORE_INDEX)
    return tokens, labels


def _encode_many(
    tokenizer: Tokenizer, rows: list[dict[str, Any]]
) -> Iterator[tuple[dict[str, Any], list[int], list[int]]]:
    """Encode message text in parallel while preserving conversation order."""
    texts = [message["content"] for row in rows for message in row["messages"]]
    encodings = iter(tokenizer.encode_batch(texts, add_special_tokens=False))
    for row in rows:
        tokens = [BOS_ID]
        labels = [IGNORE_INDEX]
        target_indices = set(row["target_message_indices"])
        for message_index, message in enumerate(row["messages"]):
            content = next(encodings).ids
            role = message["role"]
            tokens.append(ROLE_TOKENS[role])
            labels.append(IGNORE_INDEX)
            tokens.extend(content)
            supervised = message_index in target_indices
            labels.extend(content if supervised else [IGNORE_INDEX] * len(content))
            tokens.append(EOT_ID)
            supervise_eot = supervised and row.get("supervise_eot", True)
            labels.append(EOT_ID if supervise_eot else IGNORE_INDEX)
        yield row, tokens, labels


def _assistant_target_windows(
    tokens: list[int], labels: list[int], length: int
) -> list[tuple[list[int], list[int]]]:
    """Preserve every target once and give it the nearest available context."""
    if len(tokens) <= length:
        return [(tokens, labels)]
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate((*labels, IGNORE_INDEX)):
        if label != IGNORE_INDEX and start is None:
            start = index
        elif label == IGNORE_INDEX and start is not None:
            spans.append((start, index))
            start = None
    windows: list[tuple[list[int], list[int]]] = []
    # Reserve room for BOS plus at least one nearby context token whenever the
    # configured length permits it. This keeps oversized reply chunks grounded
    # in their role or immediate autoregressive prefix instead of BOS alone.
    target_capacity = max(1, length - 2)
    for span_start, span_end in spans:
        chunk_start = span_start
        while chunk_start < span_end:
            chunk_end = min(chunk_start + target_capacity, span_end)
            target_tokens = tokens[chunk_start:chunk_end]
            target_labels = labels[chunk_start:chunk_end]
            context_capacity = length - len(target_tokens)
            context = tokens[max(0, chunk_start - context_capacity) : chunk_start]
            if not context or context[0] != BOS_ID:
                context = (
                    [BOS_ID]
                    if context_capacity == 1
                    else [BOS_ID, *context[-(context_capacity - 1) :]]
                )
            window_tokens = [*context, *target_tokens]
            window_labels = [IGNORE_INDEX] * len(context) + target_labels
            if len(window_tokens) > length:
                raise SFTDataError("assistant target window exceeds sequence length")
            if not any(label != IGNORE_INDEX for label in window_labels):
                raise SFTDataError("assistant target window has no supervised labels")
            windows.append((window_tokens, window_labels))
            chunk_start = chunk_end
    if sum(label != IGNORE_INDEX for _, window in windows for label in window) != sum(
        label != IGNORE_INDEX for label in labels
    ):
        raise SFTDataError("assistant target windows do not preserve target tokens")
    return windows


def _fragments(
    tokens: list[int], labels: list[int], length: int
) -> Iterator[tuple[list[int], list[int]]]:
    start = 0
    while start < len(tokens):
        end = min(start + length, len(tokens))
        fragment_tokens = tokens[start:end]
        fragment_labels = labels[start:end].copy()
        if start:
            fragment_labels[0] = IGNORE_INDEX
        yield fragment_tokens, fragment_labels
        if end == len(tokens):
            break
        start = end - 1


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SFTDataError("SFT data config must be a mapping")
    required = {
        "schema_version",
        "dataset_id",
        "raw_root",
        "download_manifest",
        "tokenizer",
        "sequence_length",
        "validation_percent",
    }
    optional = {"windowing_strategy", "oasst_branch_policy"}
    if not required <= set(value) or set(value) - required - optional:
        raise SFTDataError(
            f"SFT data config requires {sorted(required)} and only supports "
            f"optional fields {sorted(optional)}"
        )
    if value["schema_version"] != SCHEMA_VERSION or value["validation_percent"] != 1:
        raise SFTDataError("unsupported SFT data schema or validation split")
    if not isinstance(value["sequence_length"], int) or value["sequence_length"] < 2:
        raise SFTDataError("sequence_length must be at least 2")
    if value.get("windowing_strategy", "legacy_fragments") not in {
        "legacy_fragments",
        "assistant_target_windows",
    }:
        raise SFTDataError("unsupported windowing_strategy")
    if value.get("oasst_branch_policy", "best_path") not in {
        "best_path",
        "all_assistant_paths",
    }:
        raise SFTDataError("unsupported oasst_branch_policy")
    return value


def build_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    root = Path(config["raw_root"])
    download_manifest = Path(config["download_manifest"])
    tokenizer_path = Path(config["tokenizer"])
    for path in (root, download_manifest, tokenizer_path):
        if not path.exists():
            raise SFTDataError(f"required input is missing: {path}")
    download = json.loads(download_manifest.read_text(encoding="utf-8"))
    stage8 = [item for item in download["datasets"] if 8 in item["stages"]]
    if len(stage8) != 3 or any(item["status"] != "verified" for item in stage8):
        raise SFTDataError("all three stage-8 sources must be verified")

    raw_candidates: list[dict[str, Any]] = []
    rejected = Counter()
    oasst_reader = (
        _oasst_all_assistant_paths
        if config.get("oasst_branch_policy") == "all_assistant_paths"
        else _oasst_split
    )
    sources = {
        "oasst1": oasst_reader(
            root
            / "OpenAssistant--oasst1"
            / "data"
            / "train-00000-of-00001-b42a775f407cee45.parquet",
            "oasst1",
        ),
        "ultrachat-200k": _ultrachat(root / "HuggingFaceH4--ultrachat_200k"),
        "coig": _coig(root / "BAAI--COIG"),
    }
    raw_counts: Counter[str] = Counter()
    for source, rows in sources.items():
        for row in rows:
            raw_counts[source] += 1
            raw_candidates.append(row)
    # Stable global exact deduplication. Source name is deliberately excluded.
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in sorted(
        raw_candidates, key=lambda item: (item["conversation_id"], item["source"])
    ):
        retained = deduplicated.get(row["conversation_id"])
        if retained is not None:
            retained["target_message_indices"] = sorted(
                set(retained["target_message_indices"])
                | set(row["target_message_indices"])
            )
            rejected[f"{row['source']}:exact_duplicate"] += 1
        else:
            deduplicated[row["conversation_id"]] = row
    del raw_candidates

    near_deduplicated: dict[str, dict[str, Any]] = {}
    near_keys: set[str] = set()
    for row in deduplicated.values():
        key = _near_duplicate_key(row["messages"])
        if key in near_keys:
            retained = near_deduplicated[key]
            retained["target_message_indices"] = sorted(
                set(retained["target_message_indices"])
                | set(row["target_message_indices"])
            )
            rejected[f"{row['source']}:near_duplicate"] += 1
        else:
            near_keys.add(key)
            near_deduplicated[key] = row
    del deduplicated, near_keys

    train: list[dict[str, Any]] = []
    validation_counts: Counter[str] = Counter(
        {
            "oasst1": sum(
                1
                for _ in oasst_reader(
                    root
                    / "OpenAssistant--oasst1"
                    / "data"
                    / "validation-00000-of-00001-134b8fd0c89408b6.parquet",
                    "oasst1",
                )
            ),
            "ultrachat-200k": sum(
                1
                for _ in _ultrachat(
                    root / "HuggingFaceH4--ultrachat_200k", split="test"
                )
            ),
        }
    )
    for row in near_deduplicated.values():
        # OASST and UltraChat publish held-out files. Their training files remain
        # entirely eligible; COIG gets a frozen hash split because it has none.
        if row["source"] == "coig" and int(row["conversation_id"][:8], 16) % 100 < 1:
            validation_counts[row["source"]] += 1
        else:
            train.append(row)
    train.sort(key=lambda item: item["conversation_id"])
    if not train:
        raise SFTDataError("no eligible training conversations")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequence_length = config["sequence_length"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    source_examples: Counter[str] = Counter()
    source_targets: Counter[str] = Counter()
    unique_targets = 0
    block_count = 0
    truncated_conversations = 0
    long_conversation_input_tokens = 0
    long_conversation_assistant_target_tokens = 0
    max_conversation_tokens = 0
    pii_alert_examples = 0
    anomalous_length_alert_examples = 0
    current_tokens: list[int] = []
    current_labels: list[int] = []

    def flush(inputs: Any, labels_out: Any) -> None:
        nonlocal block_count, current_tokens, current_labels
        if not current_tokens:
            return
        padding = sequence_length - len(current_tokens)
        np.asarray(current_tokens + [PAD_ID] * padding, dtype="<u4").tofile(inputs)
        np.asarray(current_labels + [IGNORE_INDEX] * padding, dtype="<i4").tofile(
            labels_out
        )
        block_count += 1
        current_tokens = []
        current_labels = []

    try:
        with (
            (temporary / INPUT_FILE).open("wb") as inputs,
            (temporary / LABEL_FILE).open("wb") as labels_out,
        ):
            for batch_start in range(0, len(train), 512):
                batch = train[batch_start : batch_start + 512]
                for row, tokens, labels in _encode_many(tokenizer, batch):
                    canonical_text = _canonical(row["messages"])
                    if _EMAIL_PATTERN.search(canonical_text) or _PHONE_PATTERN.search(
                        canonical_text
                    ):
                        pii_alert_examples += 1
                    if len(canonical_text) > 100_000 or len(row["messages"]) > 64:
                        anomalous_length_alert_examples += 1
                    target_count = sum(value != IGNORE_INDEX for value in labels)
                    if target_count == 0:
                        rejected[f"{row['source']}:no_assistant_targets"] += 1
                        continue
                    source_examples[row["source"]] += 1
                    source_targets[row["source"]] += target_count
                    unique_targets += target_count
                    max_conversation_tokens = max(max_conversation_tokens, len(tokens))
                    if len(tokens) > sequence_length:
                        truncated_conversations += 1
                        long_conversation_input_tokens += len(tokens)
                        long_conversation_assistant_target_tokens += target_count
                    windows = (
                        _assistant_target_windows(tokens, labels, sequence_length)
                        if config.get("windowing_strategy")
                        == "assistant_target_windows"
                        else list(_fragments(tokens, labels, sequence_length))
                    )
                    for window_tokens, window_labels in windows:
                        if not any(label != IGNORE_INDEX for label in window_labels):
                            if (
                                config.get("windowing_strategy")
                                == "assistant_target_windows"
                            ):
                                raise SFTDataError("v2 emitted a zero-target window")
                        if len(current_tokens) + len(window_tokens) > sequence_length:
                            flush(inputs, labels_out)
                        current_tokens.extend(window_tokens)
                        current_labels.extend(window_labels)
            flush(inputs, labels_out)
        if sum(source_examples.values()) != len(train):
            raise SFTDataError("eligible conversation accounting mismatch")
        files = {
            INPUT_FILE: {
                "bytes": (temporary / INPUT_FILE).stat().st_size,
                "sha256": _sha256(temporary / INPUT_FILE),
            },
            LABEL_FILE: {
                "bytes": (temporary / LABEL_FILE).stat().st_size,
                "sha256": _sha256(temporary / LABEL_FILE),
            },
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": config["dataset_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "config_sha256": _sha256(config_path),
            "download_manifest_sha256": _sha256(download_manifest),
            "tokenizer_sha256": _sha256(tokenizer_path),
            "sequence_length": sequence_length,
            "block_count": block_count,
            "eligible_examples": sum(source_examples.values()),
            "unique_assistant_target_tokens": unique_targets,
            "validation_examples": sum(validation_counts.values()),
            "source_normalized_candidates": dict(sorted(raw_counts.items())),
            "source_eligible_examples": dict(sorted(source_examples.items())),
            "source_assistant_target_tokens": dict(sorted(source_targets.items())),
            "source_validation_examples": dict(sorted(validation_counts.items())),
            "rejected_examples": dict(sorted(rejected.items())),
            "long_conversations_sliced": truncated_conversations,
            "long_conversation_input_tokens": long_conversation_input_tokens,
            "long_conversation_assistant_target_tokens": (
                long_conversation_assistant_target_tokens
            ),
            "max_conversation_tokens": max_conversation_tokens,
            "silent_truncation_count": 0,
            "pii_alert_examples": pii_alert_examples,
            "anomalous_length_alert_examples": anomalous_length_alert_examples,
            "near_duplicate_rule": "unicode-nfkc-casefold-remove-whitespace-and-punctuation",
            "windowing_strategy": config.get("windowing_strategy", "legacy_fragments"),
            "oasst_branch_policy": config.get("oasst_branch_policy", "best_path"),
            "packing": "stable-sha-order-no-target-truncation",
            "assistant_only_loss": True,
            "builder_sha256": _sha256(Path(__file__)),
            "formal_training_eligible": True,
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_sha = _sha256(temporary / "manifest.json")
        (temporary / "COMPLETED").write_text(
            f"manifest_sha256={manifest_sha}\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise SFTDataError(f"output already exists: {output_dir}")
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_dataset(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    completed = directory / "COMPLETED"
    if not manifest_path.is_file() or not completed.is_file():
        raise SFTDataError("SFT dataset is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        completed.read_text(encoding="utf-8")
        != f"manifest_sha256={_sha256(manifest_path)}\n"
    ):
        raise SFTDataError("SFT dataset completion marker is invalid")
    if (
        manifest.get("formal_training_eligible") is not True
        or manifest.get("assistant_only_loss") is not True
    ):
        raise SFTDataError("SFT dataset is not formally eligible")
    expected_size = manifest["block_count"] * manifest["sequence_length"] * 4
    for name in (INPUT_FILE, LABEL_FILE):
        path = directory / name
        record = manifest["files"][name]
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or record["bytes"] != expected_size
        ):
            raise SFTDataError(f"SFT data file size mismatch: {name}")
        if _sha256(path) != record["sha256"]:
            raise SFTDataError(f"SFT data file hash mismatch: {name}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the frozen stage-8 SFT dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/post_training/stage8-sft-data-v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/post-training-data/stage8-sft-v1"),
    )
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = (
        verify_dataset(args.output)
        if args.verify
        else build_dataset(args.config, args.output)
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "dataset_id",
                    "eligible_examples",
                    "unique_assistant_target_tokens",
                    "block_count",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
