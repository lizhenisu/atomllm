from pathlib import Path

import pytest
import yaml

from atomllm.data.public_pretraining_plan import (
    PublicPretrainingPlanError,
    load_plan,
)


PLAN = Path("configs/data/public-pretraining-plan-100b-v2.yaml")


def test_public_pretraining_plan_is_exactly_100b_and_fifty_ten_forty() -> None:
    plan = load_plan(PLAN)

    assert plan.total_target_tokens == 100_000_000_000
    assert plan.language_target_tokens == {
        "en": 50_000_000_000,
        "code": 10_000_000_000,
        "zh-Hans": 40_000_000_000,
    }
    assert sum(plan.source_target_tokens.values()) == plan.total_target_tokens
    assert plan.shard_token_capacity % plan.sequence_length == 0
    assert plan.token_dtype == "uint16-le"
    assert plan.source_overrides == {
        "fineweb-edu-en-score4": {"config_name": "sample-350BT"}
    }
    assert plan.training_split == "all-selected-documents"
    assert plan.validation_status == "deferred"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("synthetic_training_content", True, "synthetic"),
        ("local_text_conversion", "t2s", "conversion"),
        ("local_privacy_filtering", "regex-v1", "privacy"),
    ],
)
def test_plan_rejects_forbidden_local_processing(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    raw = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    raw[field] = value
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(PublicPretrainingPlanError, match=message):
        load_plan(path)
