"""Build a warning-free, capability-weighted cooldown token dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from atomllm.data.schema import CanonicalDocument
from atomllm.training.config import file_sha256
from atomllm.training.formal_token_shards import (
    COMPLETED_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    _write_json_atomic,
    verify_formal_token_shards,
)


class CooldownDataError(RuntimeError):
    """Raised when cooldown data violates its frozen selection contract."""


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "name",
        "split_dir",
        "split_manifest_sha256",
        "audit_manifest",
        "audit_manifest_sha256",
        "tokenizer",
        "tokenizer_sha256",
        "output_dir",
        "selection_seed",
        "require_no_quality_warnings",
        "require_no_privacy_warnings",
        "workers",
        "rules",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise CooldownDataError(f"cooldown config fields must be {sorted(required)}")
    if config["schema_version"] != 1:
        raise CooldownDataError("unsupported cooldown data schema")
    if type(config["selection_seed"]) is not int or config["selection_seed"] < 0:
        raise CooldownDataError("selection_seed must be non-negative")
    if type(config["workers"]) is not int or not 1 <= config["workers"] <= 16:
        raise CooldownDataError("workers must be in [1, 16]")
    if config["require_no_quality_warnings"] is not True:
        raise CooldownDataError("cooldown data must reject every quality warning")
    if config["require_no_privacy_warnings"] is not True:
        raise CooldownDataError("cooldown data must reject every privacy warning")
    if not isinstance(config["rules"], list) or not config["rules"]:
        raise CooldownDataError("rules must be a non-empty list")
    expected_rule_fields = {
        "name",
        "source_ids",
        "content_types",
        "languages",
        "basis_points",
    }
    names: set[str] = set()
    for rule in config["rules"]:
        if not isinstance(rule, dict) or set(rule) != expected_rule_fields:
            raise CooldownDataError(
                f"rule fields must be {sorted(expected_rule_fields)}"
            )
        if rule["name"] in names:
            raise CooldownDataError("rule names must be unique")
        names.add(rule["name"])
        if not 0 < rule["basis_points"] <= 10_000:
            raise CooldownDataError("rule basis_points must be in [1, 10000]")
        for field in ("source_ids", "content_types", "languages"):
            if not isinstance(rule[field], list) or not rule[field]:
                raise CooldownDataError(f"rule {field} must be a non-empty list")
    return config


def _selected(document: CanonicalDocument, config: dict[str, Any]) -> str | None:
    if document.quality_warnings or document.privacy_warnings:
        return None
    matches = [
        rule
        for rule in config["rules"]
        if document.source_id in rule["source_ids"]
        and document.content_type in rule["content_types"]
        and document.language in rule["languages"]
    ]
    if len(matches) > 1:
        raise CooldownDataError(f"overlapping rules for {document.document_id}")
    if not matches:
        return None
    rule = matches[0]
    digest = hashlib.sha256(
        f"{config['selection_seed']}:{document.document_id}".encode()
    ).digest()
    if int.from_bytes(digest[:8], "big") % 10_000 >= rule["basis_points"]:
        return None
    return str(rule["name"])


def _verify_lineage(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = (
        (root / config["split_dir"] / "manifest.json", config["split_manifest_sha256"]),
        (root / config["audit_manifest"], config["audit_manifest_sha256"]),
        (root / config["tokenizer"], config["tokenizer_sha256"]),
    )
    for path, expected_sha in paths:
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise CooldownDataError(f"lineage hash mismatch: {path}")
    split = json.loads(paths[0][0].read_text(encoding="utf-8"))
    audit = json.loads(paths[1][0].read_text(encoding="utf-8"))
    if audit.get("training_eligible") is not True:
        raise CooldownDataError("source audit is not training eligible")
    if audit.get("provenance", {}).get("split") != config["split_manifest_sha256"]:
        raise CooldownDataError("audit does not bind the configured split")
    shards = split.get("shards", {}).get("train")
    if not isinstance(shards, list) or not shards:
        raise CooldownDataError("split manifest has no train shards")
    return split


def _encode_worker(
    source_path: str,
    source_record: dict[str, Any],
    output_dir: str,
    output_index: int,
    tokenizer_path: str,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    path = Path(source_path)
    if file_sha256(path) != source_record["sha256"]:
        raise CooldownDataError(f"source shard hash mismatch: {path.name}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.encode_special_tokens = True
    output = Path(output_dir)
    base = f"part-{output_index:05d}"
    token_path = output / f"{base}.bin"
    index_path = output / f"{base}.idx"
    token_tmp = output / f".{base}.bin.tmp"
    index_tmp = output / f".{base}.idx.tmp"
    token_tmp.unlink(missing_ok=True)
    index_tmp.unlink(missing_ok=True)
    document_count = 0
    token_count = 0
    input_count = 0
    estimated_tokens = 0
    rule_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    try:
        with (
            path.open(encoding="utf-8") as source,
            token_tmp.open("wb") as tokens_out,
            index_tmp.open("wb") as index_out,
        ):
            for line in source:
                input_count += 1
                document = CanonicalDocument.from_json_line(line)
                rule_name = _selected(document, config)
                if rule_name is None:
                    continue
                ids = tokenizer.encode(document.text, add_special_tokens=False).ids
                if 1 in ids:
                    raise CooldownDataError("tokenizer emitted an unexpected UNK")
                encoded = np.empty(len(ids) + 2, dtype="<u2")
                encoded[0] = 2
                encoded[-1] = 3
                encoded[1:-1] = ids
                encoded.tofile(tokens_out)
                np.asarray((token_count, len(encoded)), dtype="<u8").tofile(index_out)
                token_count += len(encoded)
                document_count += 1
                estimated_tokens += int(document.metadata.get("estimated_tokens", 0))
                rule_counts[rule_name] += 1
                source_counts[document.source_id] += 1
                content_counts[document.content_type] += 1
                language_counts[document.language] += 1
            for handle in (tokens_out, index_out):
                handle.flush()
                os.fsync(handle.fileno())
        if input_count != source_record["record_count"]:
            raise CooldownDataError(f"source record count mismatch: {path.name}")
        if document_count == 0:
            token_tmp.unlink()
            index_tmp.unlink()
            return output_index, None
        os.replace(token_tmp, token_path)
        os.replace(index_tmp, index_path)
        return output_index, {
            "source_index": output_index,
            "source_name": path.name,
            "source_sha256": source_record["sha256"],
            "source_record_count": input_count,
            "document_count": document_count,
            "token_count": token_count,
            "selected_estimated_tokens": estimated_tokens,
            "rule_counts": dict(sorted(rule_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "content_counts": dict(sorted(content_counts.items())),
            "language_counts": dict(sorted(language_counts.items())),
            "token_file": {
                "name": token_path.name,
                "dtype": "uint16-le",
                "size_bytes": token_path.stat().st_size,
                "sha256": file_sha256(token_path),
            },
            "index_file": {
                "name": index_path.name,
                "dtype": "uint64-le",
                "shape": [document_count, 2],
                "size_bytes": index_path.stat().st_size,
                "sha256": file_sha256(index_path),
            },
        }
    except BaseException:
        token_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
        raise


def _sum_counters(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    result: Counter[str] = Counter()
    for item in items:
        result.update(item[field])
    return dict(sorted(result.items()))


def build(config_path: Path, project_root: Path = Path(".")) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_config(config_path)
    split = _verify_lineage(root, config)
    output = root / config["output_dir"]
    manifest_path = output / MANIFEST_NAME
    completed_path = output / COMPLETED_NAME
    if manifest_path.is_file() and completed_path.is_file():
        return verify_formal_token_shards(output)
    output.mkdir(parents=True, exist_ok=True)
    source_shards = split["shards"]["train"]
    state_path = output / "state.json"
    config_sha = file_sha256(config_path)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha:
            raise CooldownDataError("resume state config SHA-256 mismatch")
        results = state.get("shards")
        if not isinstance(results, list):
            raise CooldownDataError("resume state shards must be a list")
        for fallback_index, item in enumerate(results):
            source_index = item.setdefault("source_index", fallback_index)
            source = source_shards[source_index]
            if (
                item.get("source_name") != source["name"]
                or item.get("source_sha256") != source["sha256"]
            ):
                raise CooldownDataError("resume source lineage mismatch")
            for file_key in ("token_file", "index_file"):
                record = item[file_key]
                path = output / record["name"]
                if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                    raise CooldownDataError(f"resume file is incomplete: {path.name}")
        processed_shards = int(state.get("processed_shards", len(results)))
        if not len(results) <= processed_shards <= len(source_shards):
            raise CooldownDataError("resume processed_shards is invalid")
    else:
        results: list[dict[str, Any]] = []
        processed_shards = 0
        _write_json_atomic(
            state_path,
            {
                "schema_version": 1,
                "config_sha256": config_sha,
                "processed_shards": processed_shards,
                "shards": results,
            },
        )
    for batch_start in range(processed_shards, len(source_shards), config["workers"]):
        batch = list(
            range(batch_start, min(batch_start + config["workers"], len(source_shards)))
        )
        print(
            f"[cooldown-data] start shards {batch[0] + 1}-{batch[-1] + 1}/"
            f"{len(source_shards)}",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config["workers"]
        ) as executor:
            futures = []
            for index in batch:
                source = source_shards[index]
                source_path = (
                    root / config["split_dir"] / "train" / "shards" / source["name"]
                )
                futures.append(
                    executor.submit(
                        _encode_worker,
                        str(source_path),
                        source,
                        str(output),
                        index,
                        str(root / config["tokenizer"]),
                        config,
                    )
                )
            completed_batch = sorted(future.result() for future in futures)
        results.extend(item for _, item in completed_batch if item is not None)
        processed_shards = batch[-1] + 1
        _write_json_atomic(
            state_path,
            {
                "schema_version": 1,
                "config_sha256": config_sha,
                "processed_shards": processed_shards,
                "shards": results,
            },
        )
        print(
            f"[cooldown-data] processed {processed_shards}/{len(source_shards)} "
            f"shards, emitted {len(results)}",
            flush=True,
        )
    identity = {
        "format_version": FORMAT_VERSION,
        "name": config["name"],
        "split_manifest_sha256": config["split_manifest_sha256"],
        "audit_manifest_sha256": config["audit_manifest_sha256"],
        "tokenizer_sha256": config["tokenizer_sha256"],
        "token_dtype": "uint16-le",
        "encode_special_tokens_as_text": True,
        "selection_config_sha256": config_sha,
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "format_version": FORMAT_VERSION,
        "dataset_id": f"token-shards-{config['name']}-{identity_sha[:12]}",
        "identity_sha256": identity_sha,
        "identity": identity,
        "data_version": {
            "split_manifest_sha256": config["split_manifest_sha256"],
            "audit_manifest_sha256": config["audit_manifest_sha256"],
            "training_eligible": True,
        },
        "tokenizer": {
            "tokenizer_sha256": config["tokenizer_sha256"],
            "vocab_size": 32768,
        },
        "token_dtype": "uint16-le",
        "split": "train",
        "encode_special_tokens_as_text": True,
        "workers": config["workers"],
        "index_columns": ["token_offset", "token_count"],
        "document_count": sum(item["document_count"] for item in results),
        "token_count": sum(item["token_count"] for item in results),
        "selected_estimated_tokens": sum(
            item["selected_estimated_tokens"] for item in results
        ),
        "rule_counts": _sum_counters(results, "rule_counts"),
        "source_counts": _sum_counters(results, "source_counts"),
        "content_counts": _sum_counters(results, "content_counts"),
        "language_counts": _sum_counters(results, "language_counts"),
        "selection": {
            "seed": config["selection_seed"],
            "method": "sha256(seed:document_id)-basis-points",
            "require_no_quality_warnings": True,
            "require_no_privacy_warnings": True,
            "rules": config["rules"],
        },
        "shards": results,
        "formal_training_eligible": True,
    }
    _write_json_atomic(manifest_path, manifest)
    completed_path.write_text(
        f"{file_sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
    )
    state_path.unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build(args.config, args.project_root)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
