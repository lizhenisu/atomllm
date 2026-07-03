"""Experiment setup, reproducibility, and artifact isolation."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from atomllm.config import ProjectConfig


_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Paths and identity for one isolated experiment run."""

    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    reports_dir: Path
    config_path: Path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and all available CUDA devices."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be greater than or equal to 0")

    random.seed(seed)
    numpy_seed = np.random.SeedSequence(seed).generate_state(1)[0]
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _current_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _validate_experiment_name(name: str) -> None:
    if not _EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "experiment_name must start with an ASCII letter or digit and contain "
            "only ASCII letters, digits, dots, underscores, or hyphens"
        )
    if name in {".", ".."}:
        raise ValueError("experiment_name cannot be '.' or '..'")


def _write_config_snapshot(config: ProjectConfig, destination: Path) -> None:
    snapshot = asdict(config)
    snapshot["output_dir"] = str(config.output_dir)
    destination.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def create_run(config: ProjectConfig) -> RunContext:
    """Create a uniquely named run directory without overwriting prior artifacts."""

    _validate_experiment_name(config.experiment_name)
    run_id = f"{_current_timestamp()}_{config.experiment_name}_seed{config.seed}"
    run_dir = config.output_dir / run_id

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"run directory already exists: {run_dir}") from error

    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    checkpoints_dir.mkdir()
    logs_dir.mkdir()
    reports_dir.mkdir()

    config_path = run_dir / "config.yaml"
    _write_config_snapshot(config, config_path)

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        config_path=config_path,
    )
