"""Memory-mapped packed tokens and an exactly resumable shuffled batch cursor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from atomllm.training.packing import TOKEN_FILE_NAME, verify_packed_dataset
from atomllm.training.state import DataState


class TrainingDataError(RuntimeError):
    """Raised when packed training data or a resume cursor is incompatible."""


class PackedTokenDataset:
    """Read verified fixed-length token blocks without loading them into RAM."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest = verify_packed_dataset(self.directory)
        self.dataset_id = self.manifest["packed_data_id"]
        self.manifest_sha256 = hashlib.sha256(
            (self.directory / "manifest.json").read_bytes()
        ).hexdigest()
        self.sequence_length = self.manifest["sequence_length"]
        self.block_count = self.manifest["block_count"]
        self._tokens = np.memmap(
            self.directory / TOKEN_FILE_NAME,
            mode="r",
            dtype="<u4",
            shape=(self.block_count, self.sequence_length),
        )

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> torch.Tensor:
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        if not 0 <= index < self.block_count:
            raise IndexError("dataset index is out of range")
        return torch.from_numpy(np.array(self._tokens[index], dtype=np.int64))


class ResumableBatchIterator:
    """Yield full shuffled batches and expose the next unread sample position."""

    def __init__(
        self,
        dataset: PackedTokenDataset,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(dataset):
            raise ValueError("batch_size must not exceed the dataset")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self._order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> torch.Tensor:
        seed_bytes = f"{self.seed}\0{epoch}".encode()
        epoch_seed = int.from_bytes(
            hashlib.sha256(seed_bytes).digest()[:8],
            "little",
        )
        generator = torch.Generator()
        generator.manual_seed(epoch_seed)
        return torch.randperm(len(self.dataset), generator=generator)

    def _advance_epoch_if_needed(self) -> None:
        if self.position + self.batch_size <= len(self.dataset):
            return
        self.epoch += 1
        self.position = 0
        self._order = self._order_for_epoch(self.epoch)

    def next_batch(self) -> torch.Tensor:
        self._advance_epoch_if_needed()
        indices = self._order[self.position : self.position + self.batch_size]
        batch = torch.stack([self.dataset[int(index)] for index in indices])
        self.position += self.batch_size
        return batch

    def state(self) -> DataState:
        return DataState.from_mapping(
            {
                "format_version": 1,
                "dataset_id": self.dataset.dataset_id,
                "dataset_manifest_sha256": self.dataset.manifest_sha256,
                "split": "train",
                "epoch": self.epoch,
                "shard_index": 0,
                "shard_id": "tokens-000000",
                "sample_index": self.position,
                "token_offset": 0,
                "sampler_state": {
                    "seed": self.seed,
                    "position": self.position,
                },
            }
        )

    def restore(self, state: DataState) -> None:
        expected = {
            "dataset_id": self.dataset.dataset_id,
            "dataset_manifest_sha256": self.dataset.manifest_sha256,
            "split": "train",
            "shard_index": 0,
            "shard_id": "tokens-000000",
            "token_offset": 0,
            "seed": self.seed,
        }
        actual = {
            "dataset_id": state.dataset_id,
            "dataset_manifest_sha256": state.dataset_manifest_sha256,
            "split": state.split,
            "shard_index": state.shard_index,
            "shard_id": state.shard_id,
            "token_offset": state.token_offset,
            "seed": state.sampler_state.seed,
        }
        mismatches = [
            key
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        ]
        if mismatches:
            raise TrainingDataError(
                f"data cursor is incompatible: {', '.join(sorted(mismatches))}"
            )
        if state.sample_index != state.sampler_state.position:
            raise TrainingDataError("sample_index must equal sampler_state.position")
        if state.sample_index > len(self.dataset):
            raise TrainingDataError("data cursor position exceeds the dataset")
        if state.sample_index % self.batch_size != 0:
            raise TrainingDataError("data cursor is not on a full-batch boundary")
        self.epoch = state.epoch
        self.position = state.sample_index
        self._order = self._order_for_epoch(self.epoch)
