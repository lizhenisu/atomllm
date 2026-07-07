import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import atomllm.training.checkpoint as checkpoint_module
from atomllm.experiment import set_seed
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.checkpoint import (
    CheckpointError,
    CheckpointIdentity,
    restore_training_checkpoint,
    save_training_checkpoint,
    verify_checkpoint_directory,
)
from atomllm.training.config import (
    TrainingConfig,
    file_sha256,
    load_training_config,
)
from atomllm.training.data import PackedTokenDataset, ResumableBatchIterator
from atomllm.training.trainer import Trainer


MODEL_CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/atom-5m-baseline.yaml")


def tiny_model() -> AtomLLM:
    config = load_model_config(MODEL_CONFIG_PATH)
    config = replace(
        config,
        name="atom-checkpoint-trainer-test",
        tokenizer=replace(config.tokenizer, vocab_size=15),
        dimensions=replace(
            config.dimensions,
            max_sequence_length=4,
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            ffn_hidden_size=32,
        ),
        expected_parameter_count=2_592,
    )
    return AtomLLM(config)


def tiny_training_config() -> TrainingConfig:
    config = load_training_config(TRAINING_CONFIG_PATH)
    return replace(
        config,
        model=replace(
            config.model,
            name="atom-checkpoint-trainer-test",
            expected_parameter_count=2_592,
        ),
        batch=replace(
            config.batch,
            sequence_length=4,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        ),
        optimizer=replace(
            config.optimizer,
            learning_rate=0.01,
            weight_decay=0.0,
        ),
        scheduler=replace(
            config.scheduler,
            warmup_steps=1,
            total_steps=6,
            minimum_learning_rate_ratio=0.1,
        ),
        checkpoint=replace(
            config.checkpoint,
            save_every_steps=1,
            keep_last=2,
        ),
        runtime=replace(
            config.runtime,
            device="cpu",
            precision="fp32",
        ),
    )


def make_trainer(
    packed_dataset_dir: Path,
    model_state: dict[str, torch.Tensor],
) -> Trainer:
    config = tiny_training_config()
    model = tiny_model()
    model.load_state_dict(model_state)
    dataset = PackedTokenDataset(packed_dataset_dir)
    iterator = ResumableBatchIterator(
        dataset,
        batch_size=config.batch.micro_batch_size,
        seed=config.seed,
    )
    return Trainer(model, config, iterator)


def checkpoint_identity(run_id: str = "synthetic-run-seed42") -> CheckpointIdentity:
    return CheckpointIdentity(
        run_id=run_id,
        project_version="0.1.0",
        git_commit="0" * 40,
        git_dirty=False,
        tokenizer_sha256="a" * 64,
        config_sha256=file_sha256(TRAINING_CONFIG_PATH),
    )


def test_interrupted_training_matches_uninterrupted_training_exactly(
    tmp_path: Path,
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(301)
    initial_state = tiny_model().state_dict()
    continuous = make_trainer(packed_dataset_dir, initial_state)
    interrupted = make_trainer(packed_dataset_dir, initial_state)

    set_seed(919)
    continuous.train(4)
    expected_random = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    set_seed(919)
    interrupted.train(2)
    checkpoints_dir = tmp_path / "checkpoints"
    saved = save_training_checkpoint(
        interrupted,
        checkpoints_dir,
        checkpoint_identity(),
        keep_last=2,
    )
    assert saved.checkpoint_id == "step-000000002"
    random.random()
    np.random.random()
    torch.rand(())
    resumed = make_trainer(packed_dataset_dir, initial_state)
    restore_training_checkpoint(
        resumed,
        checkpoints_dir,
        checkpoint_identity(),
    )
    resumed.train(2)
    actual_random = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    for expected, actual in zip(
        continuous.model.parameters(),
        resumed.model.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
    assert continuous.trainer_state().global_step == 4
    assert resumed.trainer_state().global_step == 4
    assert continuous.scheduler.prepare_step() == resumed.scheduler.prepare_step()
    assert torch.equal(
        continuous.data_iterator.next_batch(),
        resumed.data_iterator.next_batch(),
    )
    assert actual_random == expected_random


def test_checkpoint_rejects_corruption_and_missing_complete(
    tmp_path: Path,
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(401)
    initial_state = tiny_model().state_dict()
    trainer = make_trainer(packed_dataset_dir, initial_state)
    trainer.train(1)
    saved = save_training_checkpoint(
        trainer,
        tmp_path / "checkpoints",
        checkpoint_identity(),
        keep_last=2,
    )
    trainer_state_path = saved.directory / "trainer_state.json"
    trainer_state_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(CheckpointError, match="size|SHA-256"):
        restore_training_checkpoint(
            make_trainer(packed_dataset_dir, initial_state),
            tmp_path / "checkpoints",
            checkpoint_identity(),
        )

    (saved.directory / "COMPLETE").unlink()
    with pytest.raises(CheckpointError, match="COMPLETE"):
        verify_checkpoint_directory(saved.directory)


def test_duplicate_step_is_rejected_and_failed_save_keeps_latest(
    tmp_path: Path,
    packed_dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(501)
    initial_state = tiny_model().state_dict()
    trainer = make_trainer(packed_dataset_dir, initial_state)
    checkpoints_dir = tmp_path / "checkpoints"
    trainer.train(1)
    save_training_checkpoint(
        trainer,
        checkpoints_dir,
        checkpoint_identity(),
        keep_last=2,
    )
    with pytest.raises(CheckpointError, match="already exists"):
        save_training_checkpoint(
            trainer,
            checkpoints_dir,
            checkpoint_identity(),
            keep_last=2,
        )

    trainer.train(1)

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated interruption"):
        save_training_checkpoint(
            trainer,
            checkpoints_dir,
            checkpoint_identity(),
            keep_last=2,
        )
    latest = json.loads((checkpoints_dir / "latest.json").read_text())
    assert latest["checkpoint_id"] == "step-000000001"
    assert not (checkpoints_dir / "step-000000002").exists()


def test_retention_preserves_recent_and_milestone_checkpoints(
    tmp_path: Path,
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(601)
    initial_state = tiny_model().state_dict()
    trainer = make_trainer(packed_dataset_dir, initial_state)
    checkpoints_dir = tmp_path / "checkpoints"

    for step in range(1, 5):
        trainer.train(1)
        save_training_checkpoint(
            trainer,
            checkpoints_dir,
            checkpoint_identity(),
            keep_last=2,
            milestone=step == 1,
        )

    checkpoint_names = sorted(
        path.name for path in checkpoints_dir.glob("step-*") if path.is_dir()
    )
    assert checkpoint_names == [
        "step-000000001",
        "step-000000003",
        "step-000000004",
    ]
    latest = json.loads((checkpoints_dir / "latest.json").read_text())
    assert latest["checkpoint_id"] == "step-000000004"


def test_restore_rejects_identity_mismatch_before_loading_state(
    tmp_path: Path,
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(701)
    initial_state = tiny_model().state_dict()
    trainer = make_trainer(packed_dataset_dir, initial_state)
    trainer.train(1)
    checkpoints_dir = tmp_path / "checkpoints"
    save_training_checkpoint(
        trainer,
        checkpoints_dir,
        checkpoint_identity(),
        keep_last=2,
    )

    incompatible = replace(checkpoint_identity(), config_sha256="b" * 64)
    with pytest.raises(CheckpointError, match="config_sha256"):
        restore_training_checkpoint(
            make_trainer(packed_dataset_dir, initial_state),
            checkpoints_dir,
            incompatible,
        )
    with pytest.raises(CheckpointError, match="invalid"):
        restore_training_checkpoint(
            make_trainer(packed_dataset_dir, initial_state),
            checkpoints_dir,
            checkpoint_identity(),
            selected_checkpoint_id="../outside",
        )
