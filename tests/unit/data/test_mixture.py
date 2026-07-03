from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from atomllm.data.mixture import (
    MixtureConfigError,
    PretrainingMixture,
    load_pretraining_mixture,
)


@pytest.fixture
def raw_mixture() -> dict[str, object]:
    return yaml.safe_load(
        Path("configs/data/pretraining-mixture.yaml").read_text(encoding="utf-8")
    )


def test_loads_default_pretraining_mixture() -> None:
    mixture = load_pretraining_mixture("configs/data/pretraining-mixture.yaml")

    assert mixture.plan_id == "atom-pretrain-v1"
    assert mixture.model_id == "Atom-Base-300M"
    assert mixture.budgets.smoke == 1_000_000
    assert mixture.budgets.main == 1_000_000_000
    assert mixture.constraints.privacy_action == "warn"


def test_default_mixture_round_trip_matches_yaml(
    raw_mixture: dict[str, object],
) -> None:
    mixture = PretrainingMixture.from_mapping(raw_mixture)

    assert mixture.to_mapping() == raw_mixture


def test_budgets_must_increase_strictly(raw_mixture: dict[str, object]) -> None:
    data = deepcopy(raw_mixture)
    data["budgets"]["pilot"] = data["budgets"]["smoke"]

    with pytest.raises(MixtureConfigError, match="must increase strictly"):
        PretrainingMixture.from_mapping(data)


@pytest.mark.parametrize("axis", ["language_mix", "content_mix", "quality_mix"])
def test_each_distribution_must_sum_to_one(
    raw_mixture: dict[str, object], axis: str
) -> None:
    data = deepcopy(raw_mixture)
    first_key = next(iter(data[axis]))
    data[axis][first_key] = 0.01

    with pytest.raises(MixtureConfigError, match=f"{axis} fractions must sum to 1"):
        PretrainingMixture.from_mapping(data)


def test_distribution_rejects_unknown_bucket(
    raw_mixture: dict[str, object],
) -> None:
    data = deepcopy(raw_mixture)
    data["language_mix"]["synthetic"] = 0.01

    with pytest.raises(MixtureConfigError, match="language_mix.*unknown fields"):
        PretrainingMixture.from_mapping(data)


def test_language_mix_must_follow_priority_order(
    raw_mixture: dict[str, object],
) -> None:
    data = deepcopy(raw_mixture)
    data["language_mix"]["zh-Hant"] = 0.07
    data["language_mix"]["ja"] = 0.10

    with pytest.raises(MixtureConfigError, match="zh-Hans > en > zh-Hant > ja"):
        PretrainingMixture.from_mapping(data)


def test_length_ranges_must_be_contiguous(raw_mixture: dict[str, object]) -> None:
    data = deepcopy(raw_mixture)
    data["length_mix"][1]["min_tokens"] = 513

    with pytest.raises(MixtureConfigError, match="ranges must be contiguous"):
        PretrainingMixture.from_mapping(data)


def test_length_mix_must_cover_native_context(
    raw_mixture: dict[str, object],
) -> None:
    data = deepcopy(raw_mixture)
    data["length_mix"][-1]["max_tokens"] = 8191

    with pytest.raises(MixtureConfigError, match="end at 8192"):
        PretrainingMixture.from_mapping(data)


def test_length_fractions_must_sum_to_one(
    raw_mixture: dict[str, object],
) -> None:
    data = deepcopy(raw_mixture)
    data["length_mix"][0]["fraction"] = 0.01

    with pytest.raises(MixtureConfigError, match="length_mix fractions must sum to 1"):
        PretrainingMixture.from_mapping(data)


def test_duplicate_limits_are_ordered(raw_mixture: dict[str, object]) -> None:
    data = deepcopy(raw_mixture)
    data["constraints"]["max_exact_duplicate_fraction"] = 0.02

    with pytest.raises(MixtureConfigError, match="cannot exceed"):
        PretrainingMixture.from_mapping(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [("privacy_action", "reject"), ("unknown_license_action", "reject")],
)
def test_warning_actions_cannot_be_changed_to_rejection(
    raw_mixture: dict[str, object], field: str, value: str
) -> None:
    data = deepcopy(raw_mixture)
    data["constraints"][field] = value

    with pytest.raises(MixtureConfigError, match=f"{field} must be 'warn'"):
        PretrainingMixture.from_mapping(data)


def test_rejects_unknown_top_level_field(raw_mixture: dict[str, object]) -> None:
    data = deepcopy(raw_mixture)
    data["unexpected"] = "synthetic"

    with pytest.raises(MixtureConfigError, match="unknown fields: unexpected"):
        PretrainingMixture.from_mapping(data)


def test_missing_mixture_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pretraining mixture not found"):
        load_pretraining_mixture(tmp_path / "missing.yaml")


def test_malformed_mixture_yaml_has_clear_error(tmp_path: Path) -> None:
    mixture_path = tmp_path / "mixture.yaml"
    mixture_path.write_text("budgets: [unterminated", encoding="utf-8")

    with pytest.raises(MixtureConfigError, match="invalid pretraining mixture YAML"):
        load_pretraining_mixture(mixture_path)
