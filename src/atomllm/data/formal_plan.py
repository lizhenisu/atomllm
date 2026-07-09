"""Stage-1 formal-data v0 source review and sampling plan validation."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from atomllm.data.mixture import PretrainingMixture, load_pretraining_mixture
from atomllm.data.schema import DataSource, SourceRegistry, load_source_registry


FORMAL_DATA_PLAN_SCHEMA_VERSION = 1
_PLAN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_VALID_STATUSES = frozenset({"draft", "approved"})
_VALID_REVIEW_STATUSES = frozenset({"pending_user_review", "approved", "rejected"})


class FormalDataPlanError(ValueError):
    """Raised when the formal-data v0 plan is invalid or unsafe to release."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormalDataPlanError(f"{context} must be a mapping with string keys")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise FormalDataPlanError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise FormalDataPlanError(f"{context} has unknown fields: {', '.join(unknown)}")


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalDataPlanError(f"{field_name} must be a non-empty string")
    return value


def _safe_relative_path(value: Any, field_name: str) -> Path:
    text = _non_empty_string(value, field_name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise FormalDataPlanError(f"{field_name} must be a safe relative path")
    return path


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise FormalDataPlanError(f"{field_name} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise FormalDataPlanError(f"{field_name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise FormalDataPlanError(f"{field_name} must not contain duplicates")
    return tuple(value)


def _fraction(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalDataPlanError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1:
        raise FormalDataPlanError(f"{field_name} must be between 0 and 1")
    return result


def _review_status(value: Any, field_name: str) -> str:
    status = _non_empty_string(value, field_name)
    if status not in _VALID_REVIEW_STATUSES:
        choices = ", ".join(sorted(_VALID_REVIEW_STATUSES))
        raise FormalDataPlanError(f"{field_name} must be one of: {choices}")
    return status


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    requires_user_license_confirmation: bool
    requires_user_terms_confirmation: bool
    allow_gated_sources_without_confirmation: bool
    privacy_action: str
    unknown_license_action: str

    @classmethod
    def from_mapping(cls, value: Any) -> ApprovalPolicy:
        data = _mapping(value, "approval_policy")
        _exact_keys(
            data,
            {
                "requires_user_license_confirmation",
                "requires_user_terms_confirmation",
                "allow_gated_sources_without_confirmation",
                "privacy_action",
                "unknown_license_action",
            },
            "approval_policy",
        )
        for field_name in (
            "requires_user_license_confirmation",
            "requires_user_terms_confirmation",
            "allow_gated_sources_without_confirmation",
        ):
            if type(data[field_name]) is not bool:
                raise FormalDataPlanError(f"approval_policy.{field_name} must be bool")
        if data["allow_gated_sources_without_confirmation"]:
            raise FormalDataPlanError(
                "approval_policy.allow_gated_sources_without_confirmation must be false"
            )
        if data["privacy_action"] != "warn":
            raise FormalDataPlanError("approval_policy.privacy_action must be 'warn'")
        if data["unknown_license_action"] != "warn":
            raise FormalDataPlanError(
                "approval_policy.unknown_license_action must be 'warn'"
            )
        return cls(
            requires_user_license_confirmation=data[
                "requires_user_license_confirmation"
            ],
            requires_user_terms_confirmation=data["requires_user_terms_confirmation"],
            allow_gated_sources_without_confirmation=False,
            privacy_action="warn",
            unknown_license_action="warn",
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "requires_user_license_confirmation": (
                self.requires_user_license_confirmation
            ),
            "requires_user_terms_confirmation": self.requires_user_terms_confirmation,
            "allow_gated_sources_without_confirmation": False,
            "privacy_action": "warn",
            "unknown_license_action": "warn",
        }


@dataclass(frozen=True, slots=True)
class SourceAllocation:
    source_id: str
    role: str
    target_language_buckets: tuple[str, ...]
    target_content_buckets: tuple[str, ...]
    max_source_fraction: float
    license_review_status: str
    terms_review_status: str
    access_review_status: str
    notes: str

    @classmethod
    def from_mapping(cls, value: Any) -> SourceAllocation:
        data = _mapping(value, "source allocation")
        _exact_keys(
            data,
            {
                "source_id",
                "role",
                "target_language_buckets",
                "target_content_buckets",
                "max_source_fraction",
                "license_review_status",
                "terms_review_status",
                "access_review_status",
                "notes",
            },
            "source allocation",
        )
        return cls(
            source_id=_non_empty_string(data["source_id"], "source_id"),
            role=_non_empty_string(data["role"], "role"),
            target_language_buckets=_string_list(
                data["target_language_buckets"], "target_language_buckets"
            ),
            target_content_buckets=_string_list(
                data["target_content_buckets"], "target_content_buckets"
            ),
            max_source_fraction=_fraction(
                data["max_source_fraction"], "max_source_fraction"
            ),
            license_review_status=_review_status(
                data["license_review_status"], "license_review_status"
            ),
            terms_review_status=_review_status(
                data["terms_review_status"], "terms_review_status"
            ),
            access_review_status=_review_status(
                data["access_review_status"], "access_review_status"
            ),
            notes=_non_empty_string(data["notes"], "notes"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "role": self.role,
            "target_language_buckets": list(self.target_language_buckets),
            "target_content_buckets": list(self.target_content_buckets),
            "max_source_fraction": self.max_source_fraction,
            "license_review_status": self.license_review_status,
            "terms_review_status": self.terms_review_status,
            "access_review_status": self.access_review_status,
            "notes": self.notes,
        }

    @property
    def user_approved(self) -> bool:
        return (
            self.license_review_status == "approved"
            and self.terms_review_status == "approved"
            and self.access_review_status == "approved"
        )


@dataclass(frozen=True, slots=True)
class KnownGap:
    bucket_type: str
    bucket: str
    reason: str

    @classmethod
    def from_mapping(cls, value: Any) -> KnownGap:
        data = _mapping(value, "known gap")
        _exact_keys(data, {"bucket_type", "bucket", "reason"}, "known gap")
        bucket_type = _non_empty_string(data["bucket_type"], "known_gaps.bucket_type")
        if bucket_type not in {"language", "content"}:
            raise FormalDataPlanError("known_gaps.bucket_type must be language/content")
        return cls(
            bucket_type=bucket_type,
            bucket=_non_empty_string(data["bucket"], "known_gaps.bucket"),
            reason=_non_empty_string(data["reason"], "known_gaps.reason"),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "bucket_type": self.bucket_type,
            "bucket": self.bucket,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FormalDataPlan:
    schema_version: int
    plan_id: str
    status: str
    training_eligible: bool
    mixture_config_path: Path
    source_registry_path: Path
    target_budget: str
    target_tokens: int
    approval_policy: ApprovalPolicy
    source_allocations: tuple[SourceAllocation, ...]
    known_gaps: tuple[KnownGap, ...]
    next_action: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        mixture: PretrainingMixture,
        registry: SourceRegistry,
    ) -> FormalDataPlan:
        data = _mapping(value, "formal data plan")
        _exact_keys(
            data,
            {
                "schema_version",
                "plan_id",
                "status",
                "training_eligible",
                "mixture_config_path",
                "source_registry_path",
                "target_budget",
                "target_tokens",
                "approval_policy",
                "source_allocations",
                "known_gaps",
                "next_action",
            },
            "formal data plan",
        )
        if data["schema_version"] != FORMAL_DATA_PLAN_SCHEMA_VERSION:
            raise FormalDataPlanError("schema_version must be 1")
        plan_id = _non_empty_string(data["plan_id"], "plan_id")
        if _PLAN_ID_PATTERN.fullmatch(plan_id) is None:
            raise FormalDataPlanError("plan_id must be lowercase path-safe")
        status = _non_empty_string(data["status"], "status")
        if status not in _VALID_STATUSES:
            raise FormalDataPlanError("status must be draft or approved")
        if type(data["training_eligible"]) is not bool:
            raise FormalDataPlanError("training_eligible must be a boolean")
        target_budget = _non_empty_string(data["target_budget"], "target_budget")
        if not hasattr(mixture.budgets, target_budget):
            raise FormalDataPlanError("target_budget is not defined in mixture")
        target_tokens = data["target_tokens"]
        if type(target_tokens) is not int or target_tokens <= 0:
            raise FormalDataPlanError("target_tokens must be a positive integer")
        expected_tokens = getattr(mixture.budgets, target_budget)
        if target_tokens != expected_tokens:
            raise FormalDataPlanError("target_tokens must match mixture budget")
        allocations = tuple(
            SourceAllocation.from_mapping(item) for item in data["source_allocations"]
        )
        gaps = tuple(KnownGap.from_mapping(item) for item in data["known_gaps"])
        plan = cls(
            schema_version=FORMAL_DATA_PLAN_SCHEMA_VERSION,
            plan_id=plan_id,
            status=status,
            training_eligible=data["training_eligible"],
            mixture_config_path=_safe_relative_path(
                data["mixture_config_path"], "mixture_config_path"
            ),
            source_registry_path=_safe_relative_path(
                data["source_registry_path"], "source_registry_path"
            ),
            target_budget=target_budget,
            target_tokens=target_tokens,
            approval_policy=ApprovalPolicy.from_mapping(data["approval_policy"]),
            source_allocations=allocations,
            known_gaps=gaps,
            next_action=_non_empty_string(data["next_action"], "next_action"),
        )
        plan._validate_against(mixture, registry)
        return plan

    def _validate_against(
        self,
        mixture: PretrainingMixture,
        registry: SourceRegistry,
    ) -> None:
        if self.status == "draft" and self.training_eligible:
            raise FormalDataPlanError("draft plans cannot be training_eligible")
        if self.status == "approved" and not self.training_eligible:
            raise FormalDataPlanError("approved plans must be training_eligible")
        if not self.source_allocations:
            raise FormalDataPlanError("source_allocations must not be empty")
        source_by_id: dict[str, DataSource] = {
            source.source_id: source for source in registry.sources
        }
        seen: set[str] = set()
        covered_languages: set[str] = set()
        covered_content: set[str] = set()
        for allocation in self.source_allocations:
            if allocation.source_id in seen:
                raise FormalDataPlanError("source_allocations source_id values unique")
            seen.add(allocation.source_id)
            source = source_by_id.get(allocation.source_id)
            if source is None:
                raise FormalDataPlanError(
                    f"source allocation is not registered: {allocation.source_id}"
                )
            if allocation.max_source_fraction > mixture.constraints.max_source_fraction:
                raise FormalDataPlanError(
                    "allocation max_source_fraction exceeds mixture constraint"
                )
            covered_languages.update(allocation.target_language_buckets)
            covered_content.update(allocation.target_content_buckets)
            if self.status == "approved" and not allocation.user_approved:
                raise FormalDataPlanError("approved plans require approved sources")
            if source.enabled and not allocation.user_approved:
                raise FormalDataPlanError("enabled sources require approved reviews")
        language_gaps = {
            gap.bucket for gap in self.known_gaps if gap.bucket_type == "language"
        }
        content_gaps = {
            gap.bucket for gap in self.known_gaps if gap.bucket_type == "content"
        }
        missing_languages = (
            set(mixture.language_mix) - covered_languages - language_gaps
        )
        missing_content = set(mixture.content_mix) - covered_content - content_gaps
        if missing_languages:
            raise FormalDataPlanError(
                f"language buckets are uncovered: {', '.join(sorted(missing_languages))}"
            )
        if missing_content:
            raise FormalDataPlanError(
                f"content buckets are uncovered: {', '.join(sorted(missing_content))}"
            )
        if self.status == "approved" and (language_gaps or content_gaps):
            raise FormalDataPlanError("approved plans cannot contain known gaps")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "status": self.status,
            "training_eligible": self.training_eligible,
            "mixture_config_path": str(self.mixture_config_path),
            "source_registry_path": str(self.source_registry_path),
            "target_budget": self.target_budget,
            "target_tokens": self.target_tokens,
            "approval_policy": self.approval_policy.to_mapping(),
            "source_allocations": [
                allocation.to_mapping() for allocation in self.source_allocations
            ],
            "known_gaps": [gap.to_mapping() for gap in self.known_gaps],
            "next_action": self.next_action,
        }


def load_formal_data_plan(
    path: str | Path, *, project_root: str | Path = "."
) -> FormalDataPlan:
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"formal data plan not found: {plan_path}")
    try:
        raw_data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise FormalDataPlanError(f"invalid formal data plan YAML: {error}") from error
    raw_mapping = _mapping(raw_data, "formal data plan")
    mixture_path = Path(project_root) / _safe_relative_path(
        raw_mapping.get("mixture_config_path"), "mixture_config_path"
    )
    registry_path = Path(project_root) / _safe_relative_path(
        raw_mapping.get("source_registry_path"), "source_registry_path"
    )
    mixture = load_pretraining_mixture(mixture_path)
    registry = load_source_registry(registry_path)
    return FormalDataPlan.from_mapping(raw_mapping, mixture=mixture, registry=registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the stage-1 formal-data v0 source review plan."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/data/formal-v0-sampling-plan.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_data_plan(args.plan, project_root=args.project_root)
    print(
        yaml.safe_dump(
            {
                "plan_id": plan.plan_id,
                "status": plan.status,
                "training_eligible": plan.training_eligible,
                "source_count": len(plan.source_allocations),
                "known_gap_count": len(plan.known_gaps),
                "next_action": plan.next_action,
            },
            sort_keys=True,
            allow_unicode=True,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
