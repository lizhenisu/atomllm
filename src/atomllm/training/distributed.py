"""Optional single-node distributed training lifecycle and collectives."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from atomllm.training.config import DistributedConfig


class DistributedError(RuntimeError):
    """Raised when the torchrun environment violates the training contract."""


def _environment_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise DistributedError(f"distributed training requires {name}")
    try:
        result = int(value)
    except ValueError as error:
        raise DistributedError(f"{name} must be an integer") from error
    if result < 0:
        raise DistributedError(f"{name} must be non-negative")
    return result


@dataclass(slots=True)
class DistributedContext:
    """Own a process group and degrade to no-op collectives for one process."""

    enabled: bool = False
    backend: str = "nccl"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    _owns_process_group: bool = False

    @classmethod
    def initialize(
        cls,
        config: DistributedConfig,
        *,
        timeout_seconds: int = 1800,
    ) -> DistributedContext:
        if not config.enabled:
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise DistributedError(
                    "torchrun world size exceeds one but distributed.enabled is false"
                )
            return cls(backend=config.backend)

        rank = _environment_integer("RANK")
        local_rank = _environment_integer("LOCAL_RANK")
        world_size = _environment_integer("WORLD_SIZE")
        if world_size < 1 or rank >= world_size:
            raise DistributedError("RANK and WORLD_SIZE are inconsistent")
        if config.backend == "nccl":
            if not torch.cuda.is_available():
                raise DistributedError("NCCL distributed training requires CUDA")
            if local_rank >= torch.cuda.device_count():
                raise DistributedError("LOCAL_RANK exceeds visible CUDA devices")
            torch.cuda.set_device(local_rank)
        if dist.is_initialized():
            raise DistributedError("a process group is already initialized")
        process_group_options = {
            "backend": config.backend,
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(seconds=timeout_seconds),
        }
        if config.backend == "nccl":
            process_group_options["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**process_group_options)
        return cls(
            enabled=True,
            backend=config.backend,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            _owns_process_group=True,
        )

    @property
    def is_distributed(self) -> bool:
        return self.enabled and self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def device(self, configured_device: str) -> torch.device:
        if configured_device == "cuda" and self.enabled:
            return torch.device("cuda", self.local_rank)
        return torch.device(configured_device)

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    def mean(self, value: float, *, device: torch.device) -> float:
        if not self.is_distributed:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor / self.world_size)

    def all_gather_object(self, value: Any) -> list[Any]:
        if not self.is_distributed:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value: Any, *, source: int = 0) -> Any:
        if not self.is_distributed:
            return value
        values = [value]
        dist.broadcast_object_list(values, src=source)
        return values[0]

    def close(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False

    def __enter__(self) -> DistributedContext:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
