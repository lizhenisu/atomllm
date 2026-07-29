import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import TrainingConfig, file_sha256, load_training_config
from atomllm.training.data import PackedTokenDataset, ResumableBatchIterator
from atomllm.training.checkpoint import CheckpointIdentity, restore_training_checkpoint
from atomllm.training.trainer import (
    Trainer,
    TrainingError,
    configure_cuda_runtime,
    train_with_checkpoints,
    write_completed_run_marker,
)


MODEL_CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/atom-5m-baseline.yaml")


def tiny_model() -> AtomLLM:
    config = load_model_config(MODEL_CONFIG_PATH)
    config = replace(
        config,
        name="atom-trainer-test",
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


def test_configure_cuda_runtime_selects_throughput_backends(monkeypatch) -> None:
    deterministic_calls = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        deterministic_calls.append,
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    configure_cuda_runtime(deterministic=False)

    assert deterministic_calls == [False]
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_completed_run_marker_binds_final_checkpoint_and_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/step-000000004"
    checkpoint.mkdir(parents=True)
    manifest = checkpoint / "manifest.json"
    manifest.write_text(
        json.dumps({"checkpoint_id": checkpoint.name, "global_step": 4}) + "\n",
        encoding="utf-8",
    )
    latest = run / "checkpoints/latest.json"
    latest.write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint.name,
                "manifest_sha256": file_sha256(manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = run / "reports/training-report-test.json"
    report.parent.mkdir()
    report.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "training.yaml"
    config.write_text("name: test\n", encoding="utf-8")

    completion = write_completed_run_marker(
        run_dir=run,
        report_path=report,
        training_config_path=config,
        final_global_step=4,
    )

    assert completion["checkpoint_id"] == "step-000000004"
    assert (run / "COMPLETED").read_text(encoding="utf-8") == (
        f"{file_sha256(run / 'completion.json')}  completion.json\n"
    )


def tiny_training_config(
    *,
    micro_batch_size: int = 2,
    accumulation_steps: int = 2,
) -> TrainingConfig:
    config = load_training_config(TRAINING_CONFIG_PATH)
    return replace(
        config,
        model=replace(
            config.model,
            name="atom-trainer-test",
            expected_parameter_count=2_592,
        ),
        batch=replace(
            config.batch,
            sequence_length=4,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=accumulation_steps,
        ),
        optimizer=replace(
            config.optimizer,
            learning_rate=0.01,
            weight_decay=0.0,
        ),
        scheduler=replace(
            config.scheduler,
            warmup_steps=1,
            total_steps=4,
            minimum_learning_rate_ratio=0.1,
        ),
        runtime=replace(
            config.runtime,
            device="cpu",
            precision="fp32",
        ),
    )


def test_trainer_updates_counters_metrics_and_data_cursor(
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(101)
    dataset = PackedTokenDataset(packed_dataset_dir)
    config = tiny_training_config()
    iterator = ResumableBatchIterator(dataset, batch_size=2, seed=config.seed)
    trainer = Trainer(tiny_model(), config, iterator)

    result = trainer.train(2)

    assert result.trainer_state.global_step == 2
    assert result.trainer_state.samples_seen == 8
    assert result.trainer_state.tokens_seen == 32
    assert result.data_state.sample_index == 8
    assert len(result.step_metrics) == 2
    assert all(metric.loss > 0 for metric in result.step_metrics)
    assert all(metric.gradient_norm > 0 for metric in result.step_metrics)
    assert result.tokens_per_second > 0


def test_trainer_uses_per_step_accumulation_schedule(
    packed_dataset_dir: Path,
) -> None:
    dataset = PackedTokenDataset(packed_dataset_dir)
    config = tiny_training_config(micro_batch_size=2, accumulation_steps=2)
    config = replace(
        config,
        batch=replace(
            config.batch,
            gradient_accumulation_schedule=(1, 2),
        ),
    )
    trainer = Trainer(
        tiny_model(),
        config,
        ResumableBatchIterator(dataset, batch_size=2, seed=config.seed),
    )

    result = trainer.train(2)

    assert result.trainer_state.samples_seen == 6
    assert result.trainer_state.tokens_seen == 24
    assert result.data_state.sample_index == 6


def test_gradient_accumulation_matches_one_combined_batch(
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(211)
    initial = tiny_model()
    accumulated_model = tiny_model()
    combined_model = tiny_model()
    accumulated_model.load_state_dict(initial.state_dict())
    combined_model.load_state_dict(initial.state_dict())
    dataset = PackedTokenDataset(packed_dataset_dir)

    accumulated_config = tiny_training_config(
        micro_batch_size=2,
        accumulation_steps=2,
    )
    combined_config = tiny_training_config(
        micro_batch_size=4,
        accumulation_steps=1,
    )
    accumulated = Trainer(
        accumulated_model,
        accumulated_config,
        ResumableBatchIterator(dataset, batch_size=2, seed=42),
    )
    combined = Trainer(
        combined_model,
        combined_config,
        ResumableBatchIterator(dataset, batch_size=4, seed=42),
    )

    accumulated.train(1)
    combined.train(1)

    for accumulated_parameter, combined_parameter in zip(
        accumulated_model.parameters(),
        combined_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            combined_parameter,
            rtol=1e-5,
            atol=1e-6,
        )


def test_trainer_rejects_runtime_and_schedule_mismatches(
    packed_dataset_dir: Path,
) -> None:
    dataset = PackedTokenDataset(packed_dataset_dir)
    config = tiny_training_config()

    with pytest.raises(TrainingError, match="batch size"):
        Trainer(
            tiny_model(),
            config,
            ResumableBatchIterator(dataset, batch_size=1, seed=42),
        )
    trainer = Trainer(
        tiny_model(),
        config,
        ResumableBatchIterator(dataset, batch_size=2, seed=42),
    )
    with pytest.raises(TrainingError, match="exceed"):
        trainer.train(5)


def test_train_with_checkpoints_saves_boundaries_and_resumes(
    tmp_path: Path,
    packed_dataset_dir: Path,
) -> None:
    torch.manual_seed(307)
    initial_state = tiny_model().state_dict()
    config = tiny_training_config()
    dataset = PackedTokenDataset(packed_dataset_dir)
    trainer = Trainer(
        tiny_model(),
        config,
        ResumableBatchIterator(dataset, batch_size=2, seed=config.seed),
    )
    trainer.model.load_state_dict(initial_state)
    identity = CheckpointIdentity(
        run_id="managed-trainer-test",
        project_version="0.1.0",
        git_commit="0" * 40,
        git_dirty=False,
        tokenizer_sha256="a" * 64,
        config_sha256=file_sha256(TRAINING_CONFIG_PATH),
    )

    observed_steps = []
    result = train_with_checkpoints(
        trainer,
        target_steps=3,
        checkpoints_dir=tmp_path / "checkpoints",
        identity=identity,
        save_every_steps=2,
        keep_last=2,
        on_step=lambda metric: observed_steps.append(metric.global_step),
    )

    assert result.trainer_state.global_step == 3
    assert observed_steps == [1, 2, 3]
    assert [event.checkpoint_id for event in result.checkpoint_events] == [
        "step-000000002",
        "step-000000003",
    ]
    assert result.checkpoint_events[-1].milestone is True

    resumed = Trainer(
        tiny_model(),
        config,
        ResumableBatchIterator(dataset, batch_size=2, seed=config.seed),
    )
    resumed.model.load_state_dict(initial_state)
    manifest = restore_training_checkpoint(
        resumed,
        tmp_path / "checkpoints",
        identity,
    )

    assert manifest["checkpoint_id"] == "step-000000003"
    assert resumed.trainer_state().global_step == 3
