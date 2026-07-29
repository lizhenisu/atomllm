"""Freeze model and training configs for the audited public 100B-token run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from atomllm.model.config import (
    ModelConfig,
    calculate_parameter_count,
    load_model_config,
)
from atomllm.tokenizer.evaluation import verify_tokenizer_directory
from atomllm.training.config import TrainingConfig
from atomllm.training.formal_token_shards import verify_formal_token_shards
from atomllm.training.public_token_shards import tokenizer_from_gpu_selection


class PublicPretrainingReleaseError(RuntimeError):
    """Raised when immutable public pretraining inputs cannot be frozen."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, path: Path, field: str) -> Path:
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not result.is_relative_to(root):
        raise PublicPretrainingReleaseError(f"{field} escapes project root")
    return result


def _model_mapping(config: ModelConfig) -> dict[str, Any]:
    return asdict(config)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def build_release(
    *,
    tokenizer_selection_dir: Path,
    token_shards_dir: Path,
    base_model_config: Path,
    output_dir: Path,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    root = project_root.resolve()
    selection_path = _resolve(root, tokenizer_selection_dir, "tokenizer_selection")
    shards_path = _resolve(root, token_shards_dir, "token_shards")
    base_model_path = _resolve(root, base_model_config, "base_model_config")
    output_path = _resolve(root, output_dir, "output_dir")
    tokenizer_path, selection_sha = tokenizer_from_gpu_selection(
        selection_path.relative_to(root), project_root=root
    )
    tokenizer, tokenizer_manifest, tokenizer_manifest_path = verify_tokenizer_directory(
        tokenizer_path
    )
    tokenizer_sha = _sha256(tokenizer_path / "tokenizer.json")
    tokenizer_manifest_sha = _sha256(tokenizer_manifest_path)
    selection = json.loads((selection_path / "report.json").read_text(encoding="utf-8"))
    if selection.get("selected_tokenizer_sha256") != tokenizer_sha:
        raise PublicPretrainingReleaseError("selected tokenizer hash mismatch")
    shards = verify_formal_token_shards(shards_path)
    identity = shards.get("identity")
    tokenizer_binding = shards.get("tokenizer")
    if not isinstance(identity, dict) or not isinstance(tokenizer_binding, dict):
        raise PublicPretrainingReleaseError("public token-shard lineage is invalid")
    if (
        identity.get("training_split") != "all-selected-documents"
        or identity.get("validation_status") != "deferred"
        or identity.get("validation_exclusion") is not None
    ):
        raise PublicPretrainingReleaseError(
            "public shards must use all selected documents for training"
        )
    expected_shard_lineage = {
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
        "gpu_selection_report_sha256": selection_sha,
    }
    mismatches = [
        key
        for key, expected in expected_shard_lineage.items()
        if identity.get(key) != expected
    ]
    if mismatches:
        raise PublicPretrainingReleaseError(
            f"token shards do not match selected tokenizer: {', '.join(mismatches)}"
        )
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_binding.get("vocab_size") != vocab_size:
        raise PublicPretrainingReleaseError("token-shard vocabulary mismatch")

    base = load_model_config(base_model_path)
    if (
        base.components.attention_dropout != 0.0
        or base.components.residual_dropout != 0.0
    ):
        raise PublicPretrainingReleaseError(
            "public base pretraining requires zero attention and residual dropout"
        )
    version_id = (
        f"tokenizer-version-atom-public-{vocab_size // 1000}k-"
        f"{tokenizer_manifest_sha[:12]}"
    )
    candidate = replace(
        base,
        name="atom-base-300m-public-v2",
        status="release",
        tokenizer=replace(
            base.tokenizer,
            version_id=version_id,
            tokenizer_sha256=tokenizer_sha,
            vocab_size=vocab_size,
        ),
    )
    model_config = replace(
        candidate,
        expected_parameter_count=calculate_parameter_count(candidate).total,
    )
    if (
        selection.get("selected_parameter_count")
        != model_config.expected_parameter_count
    ):
        raise PublicPretrainingReleaseError("GPU benchmark parameter count mismatch")

    available_samples = shards["token_count"] // shards["sequence_length"]
    world_size = 6
    micro_batch_size = 2
    accumulation_steps = 4
    global_samples_per_step = world_size * micro_batch_size * accumulation_steps
    total_steps = available_samples // global_samples_per_step
    if total_steps <= 2000:
        raise PublicPretrainingReleaseError("public dataset is unexpectedly small")
    training_samples = total_steps * global_samples_per_step
    total_tokens = training_samples * shards["sequence_length"]
    coverage = training_samples / available_samples
    if coverage < 0.99999:
        raise PublicPretrainingReleaseError("global-batch data coverage is too low")

    if output_path.exists():
        manifest_path = output_path / "manifest.json"
        completed_path = output_path / "COMPLETED"
        if manifest_path.is_file() and completed_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_shards_sha = _sha256(shards_path / "manifest.json")
            if (
                completed_path.read_text(encoding="utf-8")
                == f"{_sha256(manifest_path)}  manifest.json\n"
                and existing.get("tokenizer_selection_report_sha256") == selection_sha
                and existing.get("token_shards", {}).get("manifest_sha256")
                == current_shards_sha
                and existing.get("model_config", {}).get("vocab_size") == vocab_size
                and existing.get("model_config", {}).get("parameter_count")
                == model_config.expected_parameter_count
                and existing.get("training_config", {}).get("total_steps")
                == total_steps
                and existing.get("training_config", {}).get("expected_total_tokens")
                == total_tokens
                and existing.get("training_config", {}).get("gpu_deterministic")
                is False
            ):
                return existing
        raise PublicPretrainingReleaseError("existing release artifact is incompatible")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        model_path = temporary / "model.yaml"
        _write_yaml(model_path, _model_mapping(model_config))
        model_sha = _sha256(model_path)
        relative_model_path = (output_path / "model.yaml").relative_to(root).as_posix()
        shards_manifest_path = shards_path / "manifest.json"
        training = {
            "schema_version": 1,
            "name": "atom-base-300m-public-100b-4k-6x3090-v2",
            "status": "release",
            "seed": 20260719,
            "model": {
                "config_path": relative_model_path,
                "config_sha256": model_sha,
                "name": model_config.name,
                "expected_parameter_count": model_config.expected_parameter_count,
            },
            "data": {
                "data_version_id": shards["dataset_id"],
                "data_manifest_sha256": identity["plan_sha256"],
                "dataset_manifest_sha256": _sha256(shards_manifest_path),
                "split": "train",
                "split_sha256": shards["identity_sha256"],
                "tokenizer_version_id": version_id,
                "tokenizer_sha256": tokenizer_sha,
                "formal_training_eligible": True,
            },
            "batch": {
                "sequence_length": shards["sequence_length"],
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
            },
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.0002,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1.0e-8,
                "weight_decay": 0.1,
            },
            "scheduler": {
                "name": "trapezoidal",
                "warmup_steps": 2000,
                "total_steps": total_steps,
                "minimum_learning_rate_ratio": 0.1,
                "cooldown_steps": max(1, total_steps // 10),
            },
            "stability": {
                "max_gradient_norm": 1.0,
                "reject_non_finite_loss": True,
                "reject_non_finite_gradient_norm": True,
            },
            "checkpoint": {
                "save_every_steps": 2500,
                "keep_last": 4,
                "exact_resume": True,
                "model_format": "safetensors",
                "save_optimizer": True,
                "save_scheduler": True,
                "save_rng_state": True,
                "save_data_state": True,
            },
            "runtime": {
                "device": "cuda",
                "precision": "bf16",
                "gradient_checkpointing": False,
                "checkpoint_segment_layers": 1,
                "checkpoint_interval_segments": 1,
                "loss_chunk_size": 1024,
                "ddp_bucket_cap_mb": 50,
                "ddp_static_graph": True,
                "compile_model": False,
                "deterministic": False,
            },
            "monitoring": {
                "enabled": True,
                "tensorboard": True,
                "log_every_steps": 10,
                "flush_every_steps": 10,
            },
            "distributed": {"enabled": True, "backend": "nccl"},
            "budget": {
                "stage": "A",
                "expected_world_size": world_size,
                "expected_tokens_per_step": [
                    shards["sequence_length"]
                    * micro_batch_size
                    * accumulation_steps
                    * world_size
                ],
                "expected_training_samples": training_samples,
                "expected_total_tokens": total_tokens,
                "available_candidate_samples": available_samples,
                "minimum_coverage_ratio": 0.99999,
            },
        }
        training_path = temporary / "training.yaml"
        _write_yaml(training_path, training)
        load_model_config(model_path)
        TrainingConfig.from_mapping(training)
        release = {
            "schema_version": 1,
            "release_id": "atom-base-300m-public-100b-v2",
            "model_config": {
                "name": model_path.name,
                "sha256": model_sha,
                "parameter_count": model_config.expected_parameter_count,
                "vocab_size": vocab_size,
            },
            "training_config": {
                "name": training_path.name,
                "sha256": _sha256(training_path),
                "total_steps": total_steps,
                "expected_total_tokens": total_tokens,
                "coverage_ratio": coverage,
                "global_tokens_per_step": training["budget"][
                    "expected_tokens_per_step"
                ][0],
                "gpu_deterministic": training["runtime"]["deterministic"],
            },
            "token_shards": {
                "directory": shards_path.relative_to(root).as_posix(),
                "manifest_sha256": _sha256(shards_manifest_path),
                "dataset_id": shards["dataset_id"],
                "content_token_count": shards["content_token_count"],
                "token_count": shards["token_count"],
            },
            "validation": {
                "status": "deferred",
                "dataset": None,
            },
            "tokenizer_selection_report_sha256": selection_sha,
            "checks": {
                "full_public_100b_plan": shards["content_token_count"]
                >= 100_000_000_000,
                "language_target_contract": shards["language_content_tokens"],
                "six_gpu_training": True,
                "from_scratch": True,
                "synthetic_training_content": False,
                "model_external_capability": False,
                "all_selected_documents_are_training_candidates": True,
                "validation_deferred": True,
            },
            "training_eligible": True,
        }
        if release["checks"]["full_public_100b_plan"] is not True:
            raise PublicPretrainingReleaseError(
                "public content-token target not reached"
            )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary, output_path)
        return release
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-selection-dir", type=Path, required=True)
    parser.add_argument("--token-shards-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-config",
        type=Path,
        default=Path("configs/model/atom-base-300m.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = build_release(
        tokenizer_selection_dir=args.tokenizer_selection_dir,
        token_shards_dir=args.token_shards_dir,
        base_model_config=args.base_model_config,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
