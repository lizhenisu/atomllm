from pathlib import Path

import pytest
import yaml

from atomllm.training.final_recheck import (
    FinalRecheckError,
    load_final_recheck_config,
)


CONFIG = Path("configs/training/atom-50m-final-recheck.yaml")


def test_loads_final_process_recheck_contract() -> None:
    config = load_final_recheck_config(CONFIG)

    assert config.run_id == "stage4-atom50m-final-recheck"
    assert config.search_plan == Path("configs/training/atom-50m-experiment-plan.yaml")


def test_rejects_unsafe_final_run_id(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["run_id"] = "unsafe run"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FinalRecheckError, match="unsupported"):
        load_final_recheck_config(path)
