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
    assert config.runtime.ddp_bucket_cap_mb == 25
    assert not config.runtime.ddp_static_graph
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


@pytest.mark.parametrize(
    "path",
    [
        "configs/training/atom-base-300m-main-4k-v1.yaml",
        "configs/training/atom-base-300m-context-20k-v1.yaml",
        "configs/training/atom-base-300m-context-40k-v1.yaml",
    ],
)
def test_loads_release_ddp_performance_settings(path: str) -> None:
    config = load_training_config(path)

    assert config.runtime.ddp_bucket_cap_mb == 200
    assert config.runtime.ddp_static_graph


def test_batch_accumulation_schedule_has_exact_cumulative_budget() -> None:
    config = load_training_config(TRAINING_CONFIG)
    batch = config.batch
    scheduled = type(batch)(
        sequence_length=4096,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_accumulation_schedule=(3, 4, 4, 4),
    )

    assert [scheduled.accumulation_steps_for_step(step) for step in range(5)] == [
        3,
        4,
        4,
        4,
        3,
    ]
    assert scheduled.micro_steps_through(4) == 15
    assert scheduled.samples_through(4, 8) == 240
    assert scheduled.tokens_through(4, 8) == 983_040


def test_stage_d_cooldown_accepts_stage_c_initialization(tmp_path: Path) -> None:
    source = Path("configs/training/atom-base-300m-context-40k-v1.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["name"] = "stage-d-test"
    raw["budget"]["stage"] = "D"
    raw["initialization"] = {
        "source_stage": "C",
        "source_config_path": str(source),
        "source_config_sha256": file_sha256(source),
        "source_final_step": 1527,
        "load_optimizer_state": True,
    }
    path = tmp_path / "stage-d.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_training_config(path)

    assert config.budget is not None and config.budget.stage == "D"
    assert config.initialization is not None
    assert config.initialization.source_stage == "C"


def test_recovery_stage_accepts_stage_d_with_fresh_optimizer(tmp_path: Path) -> None:
    source = Path("configs/training/atom-base-300m-cooldown-4k-6x3090-v1.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["name"] = "recovery-test"
    raw["budget"]["stage"] = "R"
    raw["initialization"] = {
        "source_stage": "D",
        "source_config_path": str(source),
        "source_config_sha256": file_sha256(source),
        "source_final_step": 4744,
        "load_optimizer_state": False,
    }
    path = tmp_path / "recovery.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_training_config(path)

    assert config.budget is not None and config.budget.stage == "R"
    assert config.initialization is not None
    assert config.initialization.source_stage == "D"
    assert config.initialization.load_optimizer_state is False


def test_loads_trapezoidal_scheduler_with_late_cooldown(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw["scheduler"].update(
        {
            "name": "trapezoidal",
            "cooldown_steps": 100,
        }
    )
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_training_config(config_path)

    assert config.scheduler.name == "trapezoidal"
    assert config.scheduler.cooldown_steps == 100


def test_rejects_invalid_trapezoidal_cooldown(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw["scheduler"].update(
        {
            "name": "trapezoidal",
            "cooldown_steps": raw["scheduler"]["total_steps"],
        }
    )
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="warmup and cooldown"):
        load_training_config(config_path)


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


def test_rejects_nonzero_dropout_for_base_pretraining(tmp_path: Path) -> None:
    raw_model = yaml.safe_load(
        Path("configs/model/atom-micro-4m.yaml").read_text(encoding="utf-8")
    )
    raw_model["components"]["residual_dropout"] = 0.1
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(raw_model), encoding="utf-8")

    raw_training = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    raw_training["model"]["config_path"] = "model.yaml"
    raw_training["model"]["config_sha256"] = file_sha256(model_path)
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw_training), encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="base pretraining requires zero"):
        load_training_config(config_path, project_root=tmp_path)


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
