import pytest

from atomllm.training.state import (
    DataState,
    TrainerState,
    TrainingStateError,
)


TRAINER_STATE = {
    "format_version": 1,
    "global_step": 100,
    "micro_step": 0,
    "samples_seen": 3200,
    "tokens_seen": 409600,
    "optimizer_steps": 100,
    "skipped_steps": 0,
    "current_learning_rate": 0.003,
    "elapsed_training_seconds": 12.5,
}

DATA_STATE = {
    "format_version": 1,
    "dataset_id": "synthetic-smoke-v1",
    "dataset_manifest_sha256": "a" * 64,
    "split": "train",
    "epoch": 1,
    "shard_index": 2,
    "shard_id": "train-000002",
    "sample_index": 17,
    "token_offset": 0,
    "sampler_state": {"seed": 42, "position": 3217},
}


def test_trainer_state_roundtrip_and_checkpoint_boundary() -> None:
    state = TrainerState.from_mapping(TRAINER_STATE)

    assert state.global_step == 100
    assert state.to_mapping() == TRAINER_STATE

    invalid = dict(TRAINER_STATE, micro_step=1)
    with pytest.raises(TrainingStateError, match="checkpoint boundary"):
        TrainerState.from_mapping(invalid)


def test_trainer_state_requires_optimizer_step_consistency() -> None:
    invalid = dict(TRAINER_STATE, optimizer_steps=99)

    with pytest.raises(TrainingStateError, match="must equal global_step"):
        TrainerState.from_mapping(invalid)


def test_data_state_roundtrip_uses_only_logical_identity() -> None:
    state = DataState.from_mapping(DATA_STATE)

    assert state.dataset_id == "synthetic-smoke-v1"
    assert state.sampler_state.position == 3217
    assert state.to_mapping() == DATA_STATE


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_id", "/absolute/path", "logical ID"),
        ("dataset_manifest_sha256", "not-a-hash", "64 lowercase"),
        ("split", "validation", "must be 'train'"),
        ("sample_index", -1, "non-negative"),
    ],
)
def test_data_state_rejects_invalid_resume_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    invalid = dict(DATA_STATE, **{field: value})

    with pytest.raises(TrainingStateError, match=message):
        DataState.from_mapping(invalid)
