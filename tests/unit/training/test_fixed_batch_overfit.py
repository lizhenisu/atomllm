from pathlib import Path

import pytest
import yaml

from atomllm.training.fixed_batch_overfit import (
    FixedBatchOverfitError,
    load_fixed_batch_overfit_config,
)


CONFIG = Path("configs/training/atom-50m-overfit.yaml")


def test_loads_atom_50m_fixed_batch_contract() -> None:
    config = load_fixed_batch_overfit_config(CONFIG)

    assert config.sequence_length == 256
    assert config.batch_size == 1
    assert config.steps == 100
    assert config.maximum_final_to_initial_loss_ratio == 0.5


def test_rejects_non_decreasing_acceptance_ratio(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["maximum_final_to_initial_loss_ratio"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FixedBatchOverfitError, match="less than 1"):
        load_fixed_batch_overfit_config(path)
