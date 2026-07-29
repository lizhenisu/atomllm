"""Build a frozen assistant-only held-out set from official test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from atomllm.post_training.sft_data import (
    IGNORE_INDEX,
    INPUT_FILE,
    LABEL_FILE,
    PAD_ID,
    SFTDataError,
    _assistant_target_windows,
    _encode_many,
    _sha256,
)
from atomllm.post_training.sft_polish_data import (
    _deduplicate,
    _gsm8k_zh,
    _smoltalk,
    _stable_selected,
)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "dataset_id",
        "raw_root",
        "tokenizer",
        "sequence_length",
        "sample_seed",
        "smoltalk_examples",
        "final_assistant_only_sources",
        "sources",
    }
    if not isinstance(config, dict):
        raise SFTDataError("held-out config must be a mapping")
    if config.get("schema_version") == 2:
        required |= {"gsm8k_zh_split", "gsm8k_validation_basis_points"}
    if set(config) != required:
        raise SFTDataError(f"held-out config fields must be {sorted(required)}")
    if config["schema_version"] not in {1, 2}:
        raise SFTDataError("unsupported held-out data schema")
    for name in ("sequence_length", "smoltalk_examples"):
        if type(config[name]) is not int or config[name] <= 0:
            raise SFTDataError(f"{name} must be a positive integer")
    if type(config["sample_seed"]) is not int or config["sample_seed"] < 0:
        raise SFTDataError("sample_seed must be a non-negative integer")
    if config["schema_version"] == 2:
        if config["gsm8k_zh_split"] != "train":
            raise SFTDataError("schema v2 requires reserved GSM8K-zh train rows")
        if not 0 < config["gsm8k_validation_basis_points"] < 10_000:
            raise SFTDataError("invalid GSM8K-zh validation basis points")
    if set(config["sources"]) != {"smoltalk", "gsm8k_zh"}:
        raise SFTDataError("held-out sources must be SmolTalk and GSM8K-zh")
    return config


def _verify_sources(root: Path, config: dict[str, Any]) -> None:
    for source in config["sources"].values():
        directory = root / source["directory"]
        for relative, expected in source["files"].items():
            path = directory / relative
            if not path.is_file() or _sha256(path) != expected:
                raise SFTDataError(f"held-out source hash mismatch: {path}")


def _sample_key(seed: int, conversation_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{conversation_id}".encode()).digest()


def build_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    root = Path(config["raw_root"])
    tokenizer_path = Path(config["tokenizer"])
    _verify_sources(root, config)
    source_config = config["sources"]
    smoltalk = list(
        _smoltalk(
            root / source_config["smoltalk"]["directory"],
            split="test",
            final_assistant_only_sources=set(config["final_assistant_only_sources"]),
        )
    )
    if config["smoltalk_examples"] > len(smoltalk):
        raise SFTDataError("requested more SmolTalk examples than the test split")
    smoltalk.sort(
        key=lambda row: _sample_key(config["sample_seed"], row["conversation_id"])
    )
    gsm_split = config.get("gsm8k_zh_split", "test")
    gsm = list(
        _gsm8k_zh(
            root / source_config["gsm8k_zh"]["directory"] / "GSM8K_zh.json",
            split=gsm_split,
        )
    )
    if config["schema_version"] == 2:
        gsm = [
            row
            for row in gsm
            if _stable_selected(
                row["conversation_id"],
                config["gsm8k_validation_basis_points"],
                config["sample_seed"] + 1,
            )
        ]
    rows, rejected = _deduplicate([*smoltalk[: config["smoltalk_examples"]], *gsm])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequence_length = config["sequence_length"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    current_tokens: list[int] = []
    current_labels: list[int] = []
    source_examples: Counter[str] = Counter()
    source_targets: Counter[str] = Counter()
    block_count = 0
    target_tokens = 0

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
            for start in range(0, len(rows), 512):
                for row, tokens, labels in _encode_many(
                    tokenizer, rows[start : start + 512]
                ):
                    count = sum(label != IGNORE_INDEX for label in labels)
                    source_examples[row["source"]] += 1
                    source_targets[row["source"]] += count
                    target_tokens += count
                    for window_tokens, window_labels in _assistant_target_windows(
                        tokens, labels, sequence_length
                    ):
                        if len(current_tokens) + len(window_tokens) > sequence_length:
                            flush(inputs, labels_out)
                        current_tokens.extend(window_tokens)
                        current_labels.extend(window_labels)
            flush(inputs, labels_out)
        files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256(temporary / name),
            }
            for name in (INPUT_FILE, LABEL_FILE)
        }
        manifest = {
            "schema_version": 1,
            "dataset_role": "heldout-validation",
            "dataset_id": config["dataset_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "config_sha256": _sha256(config_path),
            "tokenizer_sha256": _sha256(tokenizer_path),
            "sequence_length": sequence_length,
            "block_count": block_count,
            "eligible_examples": sum(source_examples.values()),
            "unique_assistant_target_tokens": target_tokens,
            "source_eligible_examples": dict(sorted(source_examples.items())),
            "source_assistant_target_tokens": dict(sorted(source_targets.items())),
            "rejected_examples": dict(sorted(rejected.items())),
            "sample_seed": config["sample_seed"],
            "official_test_splits_only": config["schema_version"] == 1,
            "training_excluded_by_contract": config["schema_version"] == 2,
            "assistant_only_loss": True,
            "formal_training_eligible": False,
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
        raise SFTDataError("held-out dataset is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completed.read_text(encoding="utf-8") != (
        f"manifest_sha256={_sha256(manifest_path)}\n"
    ):
        raise SFTDataError("held-out completion marker is invalid")
    if (
        manifest.get("dataset_role") != "heldout-validation"
        or not (
            manifest.get("official_test_splits_only") is True
            or manifest.get("training_excluded_by_contract") is True
        )
        or manifest.get("formal_training_eligible") is not False
    ):
        raise SFTDataError("held-out dataset role is invalid")
    expected_size = manifest["block_count"] * manifest["sequence_length"] * 4
    for name in (INPUT_FILE, LABEL_FILE):
        path = directory / name
        record = manifest["files"][name]
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or record["bytes"] != expected_size
            or _sha256(path) != record["sha256"]
        ):
            raise SFTDataError(f"held-out payload mismatch: {name}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = (
        verify_dataset(args.output_dir)
        if args.verify_only
        else build_dataset(args.config, args.output_dir)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
