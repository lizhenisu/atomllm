"""Stateful learning-rate schedules with explicit optimizer-step semantics."""

from __future__ import annotations

import math
from typing import Any

import torch

from atomllm.training.config import SchedulerConfig


class SchedulerError(ValueError):
    """Raised when scheduler state or step transitions are invalid."""


class LearningRateScheduler:
    """Apply warmup plus cosine or constant learning rates before each step."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_learning_rates = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        if not self.base_learning_rates or any(
            not math.isfinite(rate) or rate <= 0 for rate in self.base_learning_rates
        ):
            raise SchedulerError("optimizer learning rates must be positive and finite")
        self.completed_steps = 0

    def multiplier(self, step_index: int) -> float:
        if type(step_index) is not int or not 0 <= step_index < self.config.total_steps:
            raise SchedulerError("step_index is outside the configured schedule")
        if self.config.warmup_steps > 0 and step_index < self.config.warmup_steps:
            return (step_index + 1) / self.config.warmup_steps
        if self.config.name == "constant":
            return 1.0
        decay_steps = self.config.total_steps - self.config.warmup_steps
        decay_index = step_index - self.config.warmup_steps
        progress = decay_index / max(decay_steps - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        minimum = self.config.minimum_learning_rate_ratio
        return minimum + (1.0 - minimum) * cosine

    def prepare_step(self) -> float:
        if self.completed_steps >= self.config.total_steps:
            raise SchedulerError("learning-rate schedule is already complete")
        multiplier = self.multiplier(self.completed_steps)
        for group, base_rate in zip(
            self.optimizer.param_groups,
            self.base_learning_rates,
            strict=True,
        ):
            group["lr"] = base_rate * multiplier
        return self.base_learning_rates[0] * multiplier

    def step(self) -> None:
        if self.completed_steps >= self.config.total_steps:
            raise SchedulerError("learning-rate schedule is already complete")
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "name": self.config.name,
            "warmup_steps": self.config.warmup_steps,
            "total_steps": self.config.total_steps,
            "minimum_learning_rate_ratio": (self.config.minimum_learning_rate_ratio),
            "base_learning_rates": list(self.base_learning_rates),
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise SchedulerError("scheduler state must be a mapping")
        expected = {
            "format_version": 1,
            "name": self.config.name,
            "warmup_steps": self.config.warmup_steps,
            "total_steps": self.config.total_steps,
            "minimum_learning_rate_ratio": (self.config.minimum_learning_rate_ratio),
            "base_learning_rates": list(self.base_learning_rates),
        }
        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise SchedulerError(f"scheduler state mismatch: {key}")
        completed_steps = state.get("completed_steps")
        if (
            type(completed_steps) is not int
            or not 0 <= completed_steps <= self.config.total_steps
        ):
            raise SchedulerError("scheduler completed_steps is invalid")
        self.completed_steps = completed_steps
