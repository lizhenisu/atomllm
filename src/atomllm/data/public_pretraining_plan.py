"""Validate the public-only 100B-token pretraining data contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from atomllm.data.public_tokenizer_corpus import (
    Source,
    load_config as load_source_registry,
)


SCHEMA_VERSION = 2
DEFAULT_PLAN = Path("configs/data/public-pretraining-plan-100b-v2.yaml")
LANGUAGES = ("en", "code", "zh-Hans")


class PublicPretrainingPlanError(RuntimeError):
    """Raised when the pretraining data plan violates its fixed contract."""


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PublicPretrainingPlanError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PublicPretrainingPlan:
    name: str
    source_registry: Path
    training_split: str
    validation_status: str
    total_target_tokens: int
    language_target_tokens: dict[str, int]
    source_target_tokens: dict[str, int]
    source_overrides: dict[str, dict[str, str | None]]
    sequence_length: int
    shard_token_capacity: int
    token_dtype: str
    document_boundary_token: str
    parent_plan_sha256: str | None = None
    supplemental_sources: tuple[Source, ...] = ()


def load_plan(
    path: str | Path = DEFAULT_PLAN,
    *,
    project_root: str | Path = ".",
) -> PublicPretrainingPlan:
    root = Path(project_root).resolve()
    plan_path = Path(path)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "name",
        "source_registry",
        "training_split",
        "validation_status",
        "total_target_tokens",
        "language_target_tokens",
        "sequence_length",
        "shard_token_capacity",
        "token_dtype",
        "document_boundary_token",
        "synthetic_training_content",
        "local_text_conversion",
        "local_privacy_filtering",
        "source_target_tokens",
        "source_overrides",
    }
    optional = {"parent_plan_sha256", "supplemental_sources"}
    if (
        not isinstance(raw, dict)
        or required - set(raw)
        or set(raw) - required - optional
    ):
        raise PublicPretrainingPlanError(
            f"plan fields must contain {sorted(required)} and optional "
            f"{sorted(optional)}"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise PublicPretrainingPlanError(f"schema_version must be {SCHEMA_VERSION}")
    if raw["synthetic_training_content"] is not False:
        raise PublicPretrainingPlanError("synthetic training content is forbidden")
    if raw["local_text_conversion"] != "none":
        raise PublicPretrainingPlanError("local Chinese conversion is forbidden")
    if raw["local_privacy_filtering"] != "none":
        raise PublicPretrainingPlanError("local privacy-pattern filtering is forbidden")
    registry_path = Path(raw["source_registry"])
    if registry_path.is_absolute() or ".." in registry_path.parts:
        raise PublicPretrainingPlanError("source_registry must be a safe relative path")
    registry = load_source_registry(root / registry_path)
    if raw["training_split"] != "all-selected-documents":
        raise PublicPretrainingPlanError(
            "training_split must be all-selected-documents"
        )
    if raw["validation_status"] != "deferred":
        raise PublicPretrainingPlanError("validation_status must be deferred")
    supplemental_raw = raw.get("supplemental_sources", [])
    if not isinstance(supplemental_raw, list):
        raise PublicPretrainingPlanError("supplemental_sources must be a list")
    supplemental_sources = tuple(
        Source.from_mapping(item) for item in supplemental_raw
    )
    all_sources = (*registry.sources, *supplemental_sources)
    source_ids = [source.source_id for source in all_sources]
    if len(source_ids) != len(set(source_ids)):
        raise PublicPretrainingPlanError(
            "base and supplemental source IDs must be unique"
        )
    sources = {source.source_id: source for source in all_sources}
    source_targets = raw["source_target_tokens"]
    if not isinstance(source_targets, dict) or set(source_targets) != set(sources):
        raise PublicPretrainingPlanError(
            "source_target_tokens must cover every fixed public source exactly once"
        )
    normalized_source_targets = {
        source_id: _positive_int(value, f"source_target_tokens.{source_id}")
        for source_id, value in source_targets.items()
    }
    raw_overrides = raw["source_overrides"]
    if not isinstance(raw_overrides, dict) or not all(
        isinstance(source_id, str) and isinstance(fields, dict)
        for source_id, fields in raw_overrides.items()
    ):
        raise PublicPretrainingPlanError("source_overrides must be a mapping")
    if not set(raw_overrides).issubset(sources):
        raise PublicPretrainingPlanError("source_overrides contains an unknown source")
    normalized_overrides: dict[str, dict[str, str | None]] = {}
    for source_id, fields in raw_overrides.items():
        if set(fields) != {"config_name"}:
            raise PublicPretrainingPlanError(
                "source override fields must contain exactly config_name"
            )
        config_name = fields["config_name"]
        if not isinstance(config_name, str) or not config_name:
            raise PublicPretrainingPlanError(
                "source override config_name must be non-empty"
            )
        if sources[source_id].data_files_pattern is not None:
            raise PublicPretrainingPlanError(
                "config_name cannot override a file-pattern source"
            )
        normalized_overrides[source_id] = {"config_name": config_name}
    language_targets = raw["language_target_tokens"]
    if not isinstance(language_targets, dict) or set(language_targets) != set(
        LANGUAGES
    ):
        raise PublicPretrainingPlanError(
            f"language_target_tokens must contain exactly {list(LANGUAGES)}"
        )
    normalized_language_targets = {
        language: _positive_int(language_targets[language], f"language.{language}")
        for language in LANGUAGES
    }
    total = _positive_int(raw["total_target_tokens"], "total_target_tokens")
    if normalized_language_targets != {
        "en": total // 2,
        "code": total // 10,
        "zh-Hans": total * 2 // 5,
    }:
        raise PublicPretrainingPlanError(
            "language token targets must be exactly 50% English, 10% code, 40% Chinese"
        )
    if sum(normalized_source_targets.values()) != total:
        raise PublicPretrainingPlanError("source token targets do not sum to total")
    actual_by_language = {language: 0 for language in LANGUAGES}
    for source_id, target in normalized_source_targets.items():
        actual_by_language[sources[source_id].language] += target
    if actual_by_language != normalized_language_targets:
        raise PublicPretrainingPlanError(
            "source token targets do not match language token targets"
        )
    sequence_length = _positive_int(raw["sequence_length"], "sequence_length")
    shard_capacity = _positive_int(raw["shard_token_capacity"], "shard_token_capacity")
    if shard_capacity % sequence_length:
        raise PublicPretrainingPlanError(
            "shard_token_capacity must be divisible by sequence_length"
        )
    if raw["token_dtype"] != "uint16-le":
        raise PublicPretrainingPlanError("token_dtype must be uint16-le")
    if raw["document_boundary_token"] != "<eos>":
        raise PublicPretrainingPlanError("document_boundary_token must be <eos>")
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise PublicPretrainingPlanError("name must be non-empty")
    parent_plan_sha256 = raw.get("parent_plan_sha256")
    if parent_plan_sha256 is not None and (
        not isinstance(parent_plan_sha256, str)
        or len(parent_plan_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in parent_plan_sha256
        )
    ):
        raise PublicPretrainingPlanError(
            "parent_plan_sha256 must be 64 lowercase hex digits or null"
        )
    return PublicPretrainingPlan(
        name=name,
        source_registry=registry_path,
        training_split="all-selected-documents",
        validation_status="deferred",
        total_target_tokens=total,
        language_target_tokens=normalized_language_targets,
        source_target_tokens=normalized_source_targets,
        source_overrides=normalized_overrides,
        sequence_length=sequence_length,
        shard_token_capacity=shard_capacity,
        token_dtype="uint16-le",
        document_boundary_token="<eos>",
        parent_plan_sha256=parent_plan_sha256,
        supplemental_sources=supplemental_sources,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    plan = load_plan(args.plan, project_root=args.project_root)
    print(
        json.dumps(
            {
                "name": plan.name,
                "total_target_tokens": plan.total_target_tokens,
                "language_target_tokens": plan.language_target_tokens,
                "source_target_tokens": plan.source_target_tokens,
                "source_overrides": plan.source_overrides,
                "training_split": plan.training_split,
                "validation_status": plan.validation_status,
                "sequence_length": plan.sequence_length,
                "shard_token_capacity": plan.shard_token_capacity,
                "parent_plan_sha256": plan.parent_plan_sha256,
                "supplemental_source_ids": [
                    source.source_id for source in plan.supplemental_sources
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
