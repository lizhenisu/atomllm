from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.training.data import (
    PackedTokenDataset,
    ResumableBatchIterator,
    TrainingDataError,
)


def test_packed_dataset_returns_independent_int64_blocks(
    packed_dataset_dir: Path,
) -> None:
    dataset = PackedTokenDataset(packed_dataset_dir)

    first = dataset[0]
    first[0] = 99

    assert len(dataset) == 10
    assert dataset.sequence_length == 4
    assert dataset[0].dtype == torch.int64
    assert dataset[0].tolist() == [2, 4, 14, 3]
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[10]


def test_restored_cursor_yields_the_exact_next_batches(
    packed_dataset_dir: Path,
) -> None:
    dataset = PackedTokenDataset(packed_dataset_dir)
    uninterrupted = ResumableBatchIterator(dataset, batch_size=2, seed=42)
    uninterrupted.next_batch()
    uninterrupted.next_batch()
    saved_state = uninterrupted.state()
    expected = [uninterrupted.next_batch() for _ in range(4)]

    resumed = ResumableBatchIterator(dataset, batch_size=2, seed=42)
    resumed.restore(saved_state)
    actual = [resumed.next_batch() for _ in range(4)]

    assert saved_state.epoch == 0
    assert saved_state.sample_index == 4
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    assert resumed.epoch == uninterrupted.epoch
    assert resumed.position == uninterrupted.position


def test_same_seed_repeats_order_and_different_seed_changes_it(
    packed_dataset_dir: Path,
) -> None:
    dataset = PackedTokenDataset(packed_dataset_dir)
    first = ResumableBatchIterator(dataset, batch_size=4, seed=7).next_batch()
    repeated = ResumableBatchIterator(dataset, batch_size=4, seed=7).next_batch()
    different = ResumableBatchIterator(dataset, batch_size=4, seed=8).next_batch()

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_cursor_rejects_incompatible_or_partial_batch_state(
    packed_dataset_dir: Path,
) -> None:
    dataset = PackedTokenDataset(packed_dataset_dir)
    iterator = ResumableBatchIterator(dataset, batch_size=2, seed=42)
    state = iterator.state()

    with pytest.raises(TrainingDataError, match="seed"):
        ResumableBatchIterator(dataset, batch_size=2, seed=43).restore(state)
    with pytest.raises(TrainingDataError, match="full-batch boundary"):
        iterator.restore(
            replace(
                state,
                sample_index=1,
                sampler_state=replace(state.sampler_state, position=1),
            )
        )
    with pytest.raises(TrainingDataError, match="sample_index"):
        iterator.restore(
            replace(
                state,
                sampler_state=replace(state.sampler_state, position=2),
            )
        )
