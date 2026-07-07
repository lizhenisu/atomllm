from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import TrainingConfig, load_training_config
from atomllm.training.data import PackedTokenDataset, ResumableBatchIterator
from atomllm.training.trainer import Trainer, TrainingError


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
