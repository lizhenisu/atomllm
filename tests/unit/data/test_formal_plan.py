from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from atomllm.data.formal_plan import (
    FormalDataPlan,
    FormalDataPlanError,
    load_formal_data_plan,
)
from atomllm.data.mixture import load_pretraining_mixture
from atomllm.data.schema import load_source_registry


PLAN_PATH = Path("configs/data/formal-v0-sampling-plan.yaml")
MIXTURE_PATH = Path("configs/data/pretraining-mixture.yaml")
REGISTRY_PATH = Path("configs/data/sources.yaml")


@pytest.fixture
def raw_plan() -> dict[str, object]:
    return yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))


def make_plan(data: dict[str, object]) -> FormalDataPlan:
    return FormalDataPlan.from_mapping(
        data,
        mixture=load_pretraining_mixture(MIXTURE_PATH),
        registry=load_source_registry(REGISTRY_PATH),
    )


def test_loads_default_formal_v0_plan_as_approved() -> None:
    plan = load_formal_data_plan(PLAN_PATH)

    assert plan.plan_id == "atom-formal-data-v0-smoke"
    assert plan.status == "approved"
    assert plan.training_eligible is True
    assert plan.target_tokens == 1_000_000
    assert len(plan.source_allocations) == 8
    assert plan.known_gaps == ()


def test_default_plan_round_trips(raw_plan: dict[str, object]) -> None:
    plan = make_plan(raw_plan)

    assert plan.to_mapping() == raw_plan


def test_draft_plan_cannot_claim_training_eligible(
    raw_plan: dict[str, object],
) -> None:
    data = deepcopy(raw_plan)
    data["status"] = "draft"
    data["training_eligible"] = True

    with pytest.raises(FormalDataPlanError, match="draft plans"):
        make_plan(data)


def test_approved_plan_requires_approved_sources(
    raw_plan: dict[str, object],
) -> None:
    data = deepcopy(raw_plan)
    data["status"] = "approved"
    data["training_eligible"] = True
    data["known_gaps"] = []
    data["source_allocations"][0]["license_review_status"] = "pending_user_review"

    with pytest.raises(FormalDataPlanError, match="approved sources"):
        make_plan(data)


def test_unknown_source_id_is_rejected(raw_plan: dict[str, object]) -> None:
    data = deepcopy(raw_plan)
    data["source_allocations"][0]["source_id"] = "missing-source-v1"

    with pytest.raises(FormalDataPlanError, match="not registered"):
        make_plan(data)


def test_source_fraction_cannot_exceed_mixture_limit(
    raw_plan: dict[str, object],
) -> None:
    data = deepcopy(raw_plan)
    data["source_allocations"][0]["max_source_fraction"] = 0.21

    with pytest.raises(FormalDataPlanError, match="exceeds mixture constraint"):
        make_plan(data)


def test_missing_bucket_requires_known_gap(raw_plan: dict[str, object]) -> None:
    data = deepcopy(raw_plan)
    for allocation in data["source_allocations"]:
        allocation["target_content_buckets"] = [
            bucket
            for bucket in allocation["target_content_buckets"]
            if bucket != "code"
        ]

    with pytest.raises(FormalDataPlanError, match="content buckets are uncovered"):
        make_plan(data)


def test_approved_plan_rejects_known_gaps(raw_plan: dict[str, object]) -> None:
    data = deepcopy(raw_plan)
    data["known_gaps"] = [
        {
            "bucket_type": "content",
            "bucket": "code",
            "reason": "synthetic gap",
        }
    ]

    with pytest.raises(FormalDataPlanError, match="cannot contain known gaps"):
        make_plan(data)


def test_gated_sources_cannot_be_allowed_without_confirmation(
    raw_plan: dict[str, object],
) -> None:
    data = deepcopy(raw_plan)
    data["approval_policy"]["allow_gated_sources_without_confirmation"] = True

    with pytest.raises(FormalDataPlanError, match="must be false"):
        make_plan(data)


def test_plan_target_tokens_must_match_budget(raw_plan: dict[str, object]) -> None:
    data = deepcopy(raw_plan)
    data["target_tokens"] = 999

    with pytest.raises(FormalDataPlanError, match="must match mixture budget"):
        make_plan(data)


def test_missing_plan_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="formal data plan not found"):
        load_formal_data_plan(tmp_path / "missing.yaml")
