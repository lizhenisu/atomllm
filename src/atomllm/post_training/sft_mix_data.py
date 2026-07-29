"""Build an audited weighted mixture from verified packed SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from atomllm.post_training.sft_data import (
    INPUT_FILE,
    LABEL_FILE,
    SFTDataError,
    verify_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema_version", "dataset_id", "sequence_length", "components"}
    if not isinstance(config, dict) or set(config) != required:
        raise SFTDataError(f"SFT mix config fields must be {sorted(required)}")
    if config["schema_version"] != 1:
        raise SFTDataError("unsupported SFT mix schema")
    if type(config["sequence_length"]) is not int or config["sequence_length"] < 2:
        raise SFTDataError("sequence_length must be at least 2")
    if not isinstance(config["components"], list) or not config["components"]:
        raise SFTDataError("components must be a non-empty list")
    names: set[str] = set()
    for component in config["components"]:
        fields = {"name", "path", "manifest_sha256", "repeat"}
        if not isinstance(component, dict) or set(component) != fields:
            raise SFTDataError(f"component fields must be {sorted(fields)}")
        if component["name"] in names:
            raise SFTDataError("component names must be unique")
        names.add(component["name"])
        if type(component["repeat"]) is not int or component["repeat"] <= 0:
            raise SFTDataError("component repeat must be a positive integer")
    return config


def build_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    sequence_length = config["sequence_length"]
    components: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for item in config["components"]:
        directory = Path(item["path"])
        manifest = verify_dataset(directory)
        if _sha256(directory / "manifest.json") != item["manifest_sha256"]:
            raise SFTDataError(f"component manifest mismatch: {item['name']}")
        if manifest["sequence_length"] != sequence_length:
            raise SFTDataError("component sequence lengths do not match")
        components.append((item, directory, manifest))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        with (
            (temporary / INPUT_FILE).open("wb") as inputs_out,
            (temporary / LABEL_FILE).open("wb") as labels_out,
        ):
            for item, directory, _ in components:
                for _ in range(item["repeat"]):
                    with (directory / INPUT_FILE).open("rb") as source:
                        shutil.copyfileobj(source, inputs_out, length=4 * 1024 * 1024)
                    with (directory / LABEL_FILE).open("rb") as source:
                        shutil.copyfileobj(source, labels_out, length=4 * 1024 * 1024)

        block_count = sum(
            item["repeat"] * manifest["block_count"] for item, _, manifest in components
        )
        unique_targets = sum(
            manifest["unique_assistant_target_tokens"] for _, _, manifest in components
        )
        packed_targets = sum(
            item["repeat"]
            * manifest.get(
                "packed_assistant_target_tokens",
                manifest["unique_assistant_target_tokens"],
            )
            for item, _, manifest in components
        )
        component_contract = {
            item["name"]: {
                "path": str(directory),
                "manifest_sha256": item["manifest_sha256"],
                "repeat": item["repeat"],
                "block_count": manifest["block_count"],
                "unique_assistant_target_tokens": manifest[
                    "unique_assistant_target_tokens"
                ],
            }
            for item, directory, manifest in components
        }
        files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256(temporary / name),
            }
            for name in (INPUT_FILE, LABEL_FILE)
        }
        manifest = {
            "schema_version": 1,
            "dataset_id": config["dataset_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "config_sha256": _sha256(config_path),
            "sequence_length": sequence_length,
            "block_count": block_count,
            "eligible_examples": sum(
                manifest["eligible_examples"] for _, _, manifest in components
            ),
            "unique_assistant_target_tokens": unique_targets,
            "packed_assistant_target_tokens": packed_targets,
            "intentional_repeated_assistant_target_tokens": (
                packed_targets - unique_targets
            ),
            "validation_examples": 0,
            "source_eligible_examples": {
                item["name"]: manifest["eligible_examples"]
                for item, _, manifest in components
            },
            "source_assistant_target_tokens": {
                item["name"]: item["repeat"]
                * manifest.get(
                    "packed_assistant_target_tokens",
                    manifest["unique_assistant_target_tokens"],
                )
                for item, _, manifest in components
            },
            "source_validation_examples": {},
            "rejected_examples": {},
            "long_conversations_sliced": 0,
            "max_conversation_tokens": sequence_length,
            "silent_truncation_count": 0,
            "packing": "verified-packed-component-mixture-v1",
            "component_contract": component_contract,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = (
        verify_dataset(args.output_dir)
        if args.verify_only
        else build_dataset(args.config, args.output_dir)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
