import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from atomllm.config import ProjectConfig
from atomllm.experiment import RunContext, create_run, set_seed


def make_config(
    tmp_path: Path, experiment_name: str = "synthetic-smoke"
) -> ProjectConfig:
    return ProjectConfig(
        experiment_name=experiment_name,
        seed=42,
        device="cuda",
        precision="bf16",
        output_dir=tmp_path / "artifacts",
    )


def sample_random_values() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def test_set_seed_reproduces_random_values() -> None:
    set_seed(42)
    first_python, first_numpy, first_torch = sample_random_values()

    set_seed(42)
    second_python, second_numpy, second_torch = sample_random_values()

    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch.equal(first_torch, second_torch)


@pytest.mark.parametrize("seed", [-1, 1.5, "42"])
def test_set_seed_rejects_invalid_values(seed: object) -> None:
    expected_error = ValueError if isinstance(seed, int) else TypeError

    with pytest.raises(expected_error):
        set_seed(seed)  # type: ignore[arg-type]


def test_create_run_builds_isolated_directory_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "atomllm.experiment._current_timestamp", lambda: "20260703-153000"
    )

    context = create_run(make_config(tmp_path))

    assert context == RunContext(
        run_id="20260703-153000_synthetic-smoke_seed42",
        run_dir=tmp_path / "artifacts/20260703-153000_synthetic-smoke_seed42",
        checkpoints_dir=(
            tmp_path / "artifacts/20260703-153000_synthetic-smoke_seed42/checkpoints"
        ),
        logs_dir=tmp_path / "artifacts/20260703-153000_synthetic-smoke_seed42/logs",
        reports_dir=(
            tmp_path / "artifacts/20260703-153000_synthetic-smoke_seed42/reports"
        ),
        config_path=(
            tmp_path / "artifacts/20260703-153000_synthetic-smoke_seed42/config.yaml"
        ),
    )
    assert context.checkpoints_dir.is_dir()
    assert context.logs_dir.is_dir()
    assert context.reports_dir.is_dir()
    assert context.config_path.is_file()


def test_create_run_snapshots_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "atomllm.experiment._current_timestamp", lambda: "20260703-153000"
    )
    config = make_config(tmp_path)

    context = create_run(config)
    snapshot = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))

    assert snapshot == {
        "experiment_name": "synthetic-smoke",
        "seed": 42,
        "device": "cuda",
        "precision": "bf16",
        "output_dir": str(tmp_path / "artifacts"),
    }


def test_create_run_refuses_to_overwrite_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "atomllm.experiment._current_timestamp", lambda: "20260703-153000"
    )
    config = make_config(tmp_path)
    first_context = create_run(config)
    marker = first_context.run_dir / "keep.txt"
    marker.write_text("synthetic marker", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run directory already exists"):
        create_run(config)

    assert marker.read_text(encoding="utf-8") == "synthetic marker"


@pytest.mark.parametrize(
    "experiment_name",
    ["../escape", "nested/name", r"nested\name", ".", "..", "/absolute"],
)
def test_create_run_rejects_unsafe_experiment_names(
    tmp_path: Path, experiment_name: str
) -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        create_run(make_config(tmp_path, experiment_name))

    assert not (tmp_path / "artifacts").exists()
