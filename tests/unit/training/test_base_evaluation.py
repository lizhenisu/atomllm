from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atomllm.training.base_evaluation import (
    BaseEvaluationError,
    _load_config,
    _sample_indices,
)
from atomllm.training.distributed import DistributedContext


def test_base_evaluation_config_loads_frozen_contract() -> None:
    config = _load_config(
        Path("configs/evaluation/atom-base-300m-stage5-heldout-v1.yaml")
    )

    assert config["sample_blocks"] == 512
    assert [item["name"] for item in config["checkpoints"]] == [
        "stage-a-4k",
        "stage-b-20k",
        "stage-c-40k",
        "sft-v2",
    ]


def test_base_evaluation_rejects_duplicate_checkpoint_names(tmp_path: Path) -> None:
    source = Path("configs/evaluation/atom-base-300m-stage5-heldout-v1.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["checkpoints"][1]["name"] = config["checkpoints"][0]["name"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(BaseEvaluationError, match="names must be unique"):
        _load_config(path)


def test_sample_indices_are_deterministic_and_unique() -> None:
    distributed = DistributedContext()

    first = _sample_indices(100, 20, 42, distributed)
    second = _sample_indices(100, 20, 42, distributed)

    assert first == second
    assert len(first) == len(set(first)) == 20
