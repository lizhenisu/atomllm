from pathlib import Path

import pytest
import yaml

from atomllm.training.config import load_training_config
from atomllm.training.sequential_search import (
    VARIABLES,
    SequentialSearchError,
    apply_candidate,
    baseline_search_settings,
    inspect_sequential_search,
    load_sequential_search_plan,
)


PLAN = Path("configs/training/atom-50m-experiment-plan.yaml")
BASE = Path("configs/training/atom-50m-baseline.yaml")


def test_formal_search_is_staged_and_not_cartesian() -> None:
    plan = load_sequential_search_plan(PLAN)

    assert {stage.variable for stage in plan.stages} == VARIABLES
    assert sum(len(stage.candidates) for stage in plan.stages) == 15
    assert plan.train_token_budget_per_trial == 1_572_864
    assert plan.final_recheck_token_budget == 6_291_456
    assert plan.validation_batches == 64
    assert plan.validation_seed == 4242
    assert plan.early_stop_check_interval_steps == 5
    assert plan.early_stop_minimum_steps == 10
    assert plan.early_stop_loss_ratio == 1.5

    summary = inspect_sequential_search(PLAN)
    assert summary["trial_steps_by_accumulation_candidate"] == {
        "16384-tokens": 96,
        "32768-tokens": 48,
        "49152-tokens": 32,
    }


def test_candidate_changes_only_named_setting() -> None:
    settings = baseline_search_settings(load_training_config(BASE))
    changed = apply_candidate(settings, "learning_rate", 0.0003)

    assert changed.learning_rate == 0.0003
    assert changed.gradient_accumulation_steps == settings.gradient_accumulation_steps
    assert changed.weight_decay == settings.weight_decay


def test_rejects_duplicate_search_variable(tmp_path: Path) -> None:
    raw = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    raw["stages"][1]["variable"] = "learning_rate"
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SequentialSearchError, match="more than once"):
        load_sequential_search_plan(path)
