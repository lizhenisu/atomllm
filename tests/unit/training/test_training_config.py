from pathlib import Path

import pytest
import yaml

from atomllm.training.config import (
    TrainingConfigError,
    file_sha256,
    load_experiment_matrix,
    load_training_config,
)


TRAINING_CONFIG = Path("configs/training/atom-5m-baseline.yaml")
MATRIX_CONFIG = Path("configs/training/atom-5m-matrix.yaml")


def test_loads_bound_atom_5m_baseline() -> None:
    config = load_training_config(TRAINING_CONFIG)

    assert config.name == "atom-5m-stage4-baseline"
    assert config.status == "smoke"
    assert config.model.expected_parameter_count == 4_465_280
    assert config.batch.tokens_per_micro_batch == 1_024
    assert config.batch.tokens_per_optimizer_step == 4_096
    assert not config.data.formal_training_eligible
    assert config.checkpoint.exact_resume
    assert config.runtime.loss_chunk_size is None
    assert config.monitoring.enabled
    assert config.monitoring.tensorboard
    assert config.monitoring.log_every_steps == 1
    assert not config.distributed.enabled
    assert config.distributed.backend == "nccl"


def test_loads_distributed_training_config() -> None:
    config = load_training_config("configs/training/atom-base-300m-long-6x3090-v1.yaml")

    assert config.distributed.enabled
    assert config.distributed.backend == "nccl"
    assert config.scheduler.total_steps == 61865
    assert config.scheduler.warmup_steps == 816
    assert config.checkpoint.save_every_steps == 50


def test_loads_explicit_one_variable_atom_5m_matrix() -> None:
    matrix, base = load_experiment_matrix(MATRIX_CONFIG)

    assert matrix.name == "atom-5m-stage4-matrix-v1"
    assert len(matrix.trials) == 8
    assert matrix.trials[0].name == "baseline"
    assert base.scheduler.total_steps == 500
    assert matrix.trials[-1].gradient_accumulation_steps == 8


def test_rejects_model_hash_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw["model"]["config_sha256"] = "0" * 64
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="model config SHA-256"):
        load_training_config(config_path)


def test_rejects_release_status_with_smoke_data(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw["status"] = "release"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="formally eligible"):
        load_training_config(config_path)


def test_rejects_unsafe_model_path_and_unknown_field(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw["model"]["config_path"] = "../outside.yaml"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="safe relative path"):
        load_training_config(config_path)

    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw["optimizer"]["unexpected"] = True
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="unknown fields"):
        load_training_config(config_path)


def test_matrix_base_hash_matches_checked_in_baseline() -> None:
    raw = yaml.safe_load(MATRIX_CONFIG.read_text(encoding="utf-8"))

    assert raw["base_config_sha256"] == file_sha256(TRAINING_CONFIG)
