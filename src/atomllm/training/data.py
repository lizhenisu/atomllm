"""Memory-mapped packed tokens and an exactly resumable shuffled batch cursor."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch

from atomllm.training.packing import TOKEN_FILE_NAME, verify_packed_dataset
from atomllm.training.formal_token_shards import verify_formal_token_shards
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
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(dataset):
            raise ValueError("batch_size must not exceed the dataset")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        if batch_size * world_size > len(dataset):
            raise ValueError("global batch size must not exceed the dataset")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.global_batch_size = batch_size * world_size
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
        if self.position + self.global_batch_size <= len(self.dataset):
            return
        self.epoch += 1
        self.position = 0
        self._order = self._order_for_epoch(self.epoch)

    def next_batch(self) -> torch.Tensor:
        self._advance_epoch_if_needed()
        start = self.position + self.rank * self.batch_size
        indices = self._order[start : start + self.batch_size]
        batch = torch.stack([self.dataset[int(index)] for index in indices])
        self.position += self.global_batch_size
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
        if state.sample_index % self.global_batch_size != 0:
            raise TrainingDataError(
                "data cursor is not on a full-batch boundary for the global batch"
            )
        self.epoch = state.epoch
        self.position = state.sample_index
        self._order = self._order_for_epoch(self.epoch)


class ShardedTokenDataset:
    """Expose fixed-length samples over uint16 token shards with bounded RAM."""

    def __init__(
        self,
        directory: str | Path,
        *,
        sequence_length: int,
        verified_manifest: dict[str, Any] | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        if type(sequence_length) is not int or sequence_length < 2:
            raise ValueError("sequence_length must be an integer of at least 2")
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if verified_manifest is None:
            if manifest_sha256 is not None:
                raise ValueError("manifest_sha256 requires a verified_manifest")
            self.manifest = verify_formal_token_shards(self.directory)
        else:
            if not isinstance(manifest_sha256, str):
                raise ValueError("verified_manifest requires manifest_sha256")
            actual_manifest_sha256 = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            if actual_manifest_sha256 != manifest_sha256:
                raise TrainingDataError(
                    "broadcast token-shard manifest does not match local storage"
                )
            self.manifest = verified_manifest
        self.dataset_id = self.manifest["dataset_id"]
        self.manifest_sha256 = (
            manifest_sha256
            if manifest_sha256 is not None
            else hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )
        self.sequence_length = sequence_length
        self._tokens: list[np.memmap] = []
        self._block_counts: list[int] = []
        self._cumulative_blocks: list[int] = []
        cumulative = 0
        for item in self.manifest["shards"]:
            token_count = item["token_count"]
            block_count = token_count // sequence_length
            token_path = self.directory / item["token_file"]["name"]
            if (
                not token_path.is_file()
                or token_path.stat().st_size != item["token_file"]["size_bytes"]
            ):
                raise TrainingDataError(
                    f"token shard is missing or truncated: {token_path.name}"
                )
            self._tokens.append(
                np.memmap(
                    token_path,
                    mode="r",
                    dtype="<u2",
                    shape=(token_count,),
                )
            )
            self._block_counts.append(block_count)
            cumulative += block_count
            self._cumulative_blocks.append(cumulative)
        self.block_count = cumulative
        if self.block_count == 0:
            raise TrainingDataError("token shards contain no complete sequence")

    def __len__(self) -> int:
        return self.block_count

    def locate(self, index: int) -> tuple[int, int]:
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        if not 0 <= index < self.block_count:
            raise IndexError("dataset index is out of range")
        shard_index = bisect_right(self._cumulative_blocks, index)
        previous = 0 if shard_index == 0 else self._cumulative_blocks[shard_index - 1]
        return shard_index, index - previous

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_index, local_block = self.locate(index)
        start = local_block * self.sequence_length
        tokens = self._tokens[shard_index][start : start + self.sequence_length]
        return torch.from_numpy(np.array(tokens, dtype=np.int64))


class ResumableShardedBatchIterator:
    """O(1)-memory deterministic permutation and resumable batch cursor."""

    def __init__(
        self,
        dataset: ShardedTokenDataset,
        *,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(dataset):
            raise ValueError("batch_size must not exceed the dataset")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        if batch_size * world_size > len(dataset):
            raise ValueError("global batch size must not exceed the dataset")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.global_batch_size = batch_size * world_size
        self.epoch = 0
        self.position = 0
        self._multiplier, self._offset = self._permutation_for_epoch(0)

    def _permutation_for_epoch(self, epoch: int) -> tuple[int, int]:
        digest = hashlib.sha256(f"{self.seed}\0{epoch}".encode()).digest()
        multiplier = int.from_bytes(digest[:8], "little") % len(self.dataset)
        if multiplier == 0:
            multiplier = 1
        while math.gcd(multiplier, len(self.dataset)) != 1:
            multiplier = (multiplier + 1) % len(self.dataset)
            if multiplier == 0:
                multiplier = 1
        offset = int.from_bytes(digest[8:16], "little") % len(self.dataset)
        return multiplier, offset

    def _dataset_index(self, position: int) -> int:
        return (self._multiplier * position + self._offset) % len(self.dataset)

    def _advance_epoch_if_needed(self) -> None:
        if self.position + self.global_batch_size <= len(self.dataset):
            return
        self.epoch += 1
        self.position = 0
        self._multiplier, self._offset = self._permutation_for_epoch(self.epoch)

    def next_batch(self) -> torch.Tensor:
        self._advance_epoch_if_needed()
        indices = [
            self._dataset_index(position)
            for position in range(
                self.position + self.rank * self.batch_size,
                self.position + (self.rank + 1) * self.batch_size,
            )
        ]
        batch = torch.stack([self.dataset[index] for index in indices])
        self.position += self.global_batch_size
        return batch

    def _next_location(self) -> tuple[int, int, int]:
        epoch = self.epoch
        position = self.position
        if position + self.global_batch_size > len(self.dataset):
            epoch += 1
            position = 0
            multiplier, offset = self._permutation_for_epoch(epoch)
            rank_position = position + self.rank * self.batch_size
            dataset_index = (multiplier * rank_position + offset) % len(self.dataset)
        else:
            dataset_index = self._dataset_index(position + self.rank * self.batch_size)
        shard_index, local_block = self.dataset.locate(dataset_index)
        return shard_index, local_block, epoch

    def state(self) -> DataState:
        shard_index, local_block, next_epoch = self._next_location()
        return DataState.from_mapping(
            {
                "format_version": 1,
                "dataset_id": self.dataset.dataset_id,
                "dataset_manifest_sha256": self.dataset.manifest_sha256,
                "split": "train",
                "epoch": next_epoch,
                "shard_index": shard_index,
                "shard_id": self.dataset.manifest["shards"][shard_index]["token_file"][
                    "name"
                ],
                "sample_index": 0 if next_epoch != self.epoch else self.position,
                "token_offset": local_block * self.dataset.sequence_length,
                "sampler_state": {
                    "seed": self.seed,
                    "position": 0 if next_epoch != self.epoch else self.position,
                },
            }
        )

    def restore(self, state: DataState) -> None:
        if state.dataset_id != self.dataset.dataset_id:
            raise TrainingDataError("data cursor dataset_id is incompatible")
        if state.dataset_manifest_sha256 != self.dataset.manifest_sha256:
            raise TrainingDataError("data cursor manifest is incompatible")
        if state.split != "train" or state.sampler_state.seed != self.seed:
            raise TrainingDataError("data cursor split or seed is incompatible")
        if state.sample_index != state.sampler_state.position:
            raise TrainingDataError("sample_index must equal sampler_state.position")
        if state.sample_index > len(self.dataset):
            raise TrainingDataError("data cursor position exceeds the dataset")
        if state.sample_index % self.global_batch_size != 0:
            raise TrainingDataError(
                "data cursor is not on a full-batch boundary for the global batch"
            )
        self.epoch = state.epoch
        self.position = state.sample_index
        self._multiplier, self._offset = self._permutation_for_epoch(self.epoch)
        expected = self.state()
        for field in ("shard_index", "shard_id", "token_offset"):
            if getattr(state, field) != getattr(expected, field):
                raise TrainingDataError(f"data cursor {field} is incompatible")
