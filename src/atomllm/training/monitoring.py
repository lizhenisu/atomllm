"""Durable console, JSONL, and TensorBoard monitoring for training runs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    global_step: int
    total_steps: int
    loss: float
    gradient_norm: float
    learning_rate: float
    samples_seen: int
    tokens_seen: int
    tokens_per_second: float
    step_seconds: float
    elapsed_seconds: float
    eta_seconds: float
    gpu_allocated_gib: float
    gpu_reserved_gib: float


class TrainingMonitor:
    """Append recoverable progress without participating in training state."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        total_steps: int,
        tokens_per_step: int,
        start_step: int,
        total_tokens: int | None = None,
        start_tokens: int = 0,
        prior_elapsed_seconds: float = 0.0,
        log_every_steps: int = 1,
        flush_every_steps: int = 1,
        tensorboard: bool = True,
    ) -> None:
        if total_steps <= 0 or tokens_per_step <= 0:
            raise ValueError("monitoring totals must be positive")
        if not 0 <= start_step <= total_steps:
            raise ValueError("monitoring start_step is outside the schedule")
        if total_tokens is not None and total_tokens <= 0:
            raise ValueError("monitoring total_tokens must be positive")
        if start_tokens < 0 or (
            total_tokens is not None and start_tokens > total_tokens
        ):
            raise ValueError("monitoring start_tokens is outside the budget")
        if log_every_steps <= 0 or flush_every_steps <= 0:
            raise ValueError("monitoring intervals must be positive")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.tokens_per_step = tokens_per_step
        self.start_step = start_step
        self.total_tokens = total_tokens
        self.start_tokens = start_tokens
        self.prior_elapsed_seconds = prior_elapsed_seconds
        self.log_every_steps = log_every_steps
        self.flush_every_steps = flush_every_steps
        self.started = time.perf_counter()
        self.last_event_at = self.started
        self.last_step = start_step
        self._events_since_flush = 0
        self._jsonl: TextIO = (self.log_dir / "progress.jsonl").open(
            "a", encoding="utf-8"
        )
        self._writer = (
            SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            if tensorboard
            else None
        )

    def record(self, metric: object) -> ProgressEvent | None:
        step = int(getattr(metric, "global_step"))
        if step % self.log_every_steps and step != self.total_steps:
            return None
        now = time.perf_counter()
        advanced_steps = step - self.last_step
        step_seconds = (now - self.last_event_at) / max(advanced_steps, 1)
        process_elapsed = now - self.started
        elapsed = self.prior_elapsed_seconds + process_elapsed
        tokens_seen = int(getattr(metric, "tokens_seen"))
        process_tokens = max(tokens_seen - self.start_tokens, 0)
        tokens_per_second = process_tokens / max(process_elapsed, 1e-12)
        if self.total_tokens is None:
            remaining_tokens = max(self.total_steps - step, 0) * self.tokens_per_step
        else:
            remaining_tokens = max(self.total_tokens - tokens_seen, 0)
        eta_seconds = remaining_tokens / max(tokens_per_second, 1e-12)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            allocated = reserved = 0.0
        event = ProgressEvent(
            global_step=step,
            total_steps=self.total_steps,
            loss=float(getattr(metric, "loss")),
            gradient_norm=float(getattr(metric, "gradient_norm")),
            learning_rate=float(getattr(metric, "learning_rate")),
            samples_seen=int(getattr(metric, "samples_seen")),
            tokens_seen=tokens_seen,
            tokens_per_second=tokens_per_second,
            step_seconds=step_seconds,
            elapsed_seconds=elapsed,
            eta_seconds=eta_seconds,
            gpu_allocated_gib=allocated,
            gpu_reserved_gib=reserved,
        )
        if not all(
            math.isfinite(value)
            for value in (
                event.loss,
                event.gradient_norm,
                event.learning_rate,
                event.tokens_per_second,
                event.step_seconds,
                event.elapsed_seconds,
                event.eta_seconds,
                event.gpu_allocated_gib,
                event.gpu_reserved_gib,
            )
        ):
            raise ValueError("monitoring event contains a non-finite value")
        self._jsonl.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        print(
            "[train] "
            f"step={step}/{self.total_steps} "
            f"loss={event.loss:.6f} grad={event.gradient_norm:.4f} "
            f"lr={event.learning_rate:.3e} tok/s={event.tokens_per_second:.2f} "
            f"vram={event.gpu_allocated_gib:.3f}GiB "
            f"eta={event.eta_seconds / 3600:.1f}h",
            flush=True,
        )
        if self._writer is not None:
            scalars = {
                "train/loss": event.loss,
                "train/gradient_norm": event.gradient_norm,
                "train/learning_rate": event.learning_rate,
                "train/tokens_per_second": event.tokens_per_second,
                "train/tokens_seen": event.tokens_seen,
                "system/gpu_allocated_gib": event.gpu_allocated_gib,
                "system/gpu_reserved_gib": event.gpu_reserved_gib,
                "system/step_seconds": event.step_seconds,
                "system/elapsed_hours": event.elapsed_seconds / 3600,
                "system/eta_hours": event.eta_seconds / 3600,
            }
            for tag, value in scalars.items():
                self._writer.add_scalar(tag, value, step)
        self.last_event_at = now
        self.last_step = step
        self._events_since_flush += 1
        if self._events_since_flush >= self.flush_every_steps:
            self.flush()
        return event

    def flush(self) -> None:
        self._jsonl.flush()
        if self._writer is not None:
            self._writer.flush()
        self._events_since_flush = 0

    def close(self) -> None:
        self.flush()
        self._jsonl.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> TrainingMonitor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
