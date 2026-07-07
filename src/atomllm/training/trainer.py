"""AtomLLM gradient-accumulation trainer for stage-4 scale experiments."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ContextManager

import torch

from atomllm.config import ProjectConfig
from atomllm.experiment import create_run, set_seed
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import TrainingConfig, load_training_config
from atomllm.training.data import PackedTokenDataset, ResumableBatchIterator
from atomllm.training.scheduler import LearningRateScheduler
from atomllm.training.state import DataState, TrainerState


class TrainingError(RuntimeError):
    """Raised when a training step violates stability or runtime requirements."""


@dataclass(frozen=True, slots=True)
class StepMetrics:
    global_step: int
    loss: float
    gradient_norm: float
    learning_rate: float
    samples_seen: int
    tokens_seen: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    trainer_state: TrainerState
    data_state: DataState
    step_metrics: tuple[StepMetrics, ...]
    peak_allocated_gib: float
    peak_reserved_gib: float
    tokens_per_second: float


def build_adamw_optimizer(
    model: AtomLLM,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Use weight decay for matrices and exclude one-dimensional RMSNorm weights."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not decay or not no_decay:
        raise TrainingError("optimizer parameter groups are incomplete")
    optimizer = config.optimizer
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": optimizer.weight_decay,
                "group_name": "decay",
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "group_name": "no-decay",
            },
        ],
        lr=optimizer.learning_rate,
        betas=(optimizer.beta1, optimizer.beta2),
        eps=optimizer.epsilon,
    )


class Trainer:
    """Own model, optimizer, schedule, counters, and the next unread data cursor."""

    def __init__(
        self,
        model: AtomLLM,
        config: TrainingConfig,
        data_iterator: ResumableBatchIterator,
    ) -> None:
        if config.runtime.gradient_checkpointing:
            raise TrainingError("gradient checkpointing is not implemented yet")
        if config.runtime.compile_model:
            raise TrainingError("compile_model is not implemented yet")
        if config.runtime.device == "cuda" and not torch.cuda.is_available():
            raise TrainingError("CUDA training was requested but is unavailable")
        if data_iterator.batch_size != config.batch.micro_batch_size:
            raise TrainingError("data iterator batch size does not match config")
        if data_iterator.dataset.sequence_length != config.batch.sequence_length:
            raise TrainingError("packed sequence length does not match config")

        self.config = config
        self.device = torch.device(config.runtime.device)
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.data_iterator = data_iterator
        self.optimizer = build_adamw_optimizer(self.model, config)
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            config.scheduler,
        )
        self.samples_seen = 0
        self.tokens_seen = 0
        self.skipped_steps = 0
        self.elapsed_training_seconds = 0.0
        self._last_learning_rate = 0.0

    def _autocast(self) -> ContextManager[None]:
        if self.config.runtime.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def trainer_state(self) -> TrainerState:
        return TrainerState.from_mapping(
            {
                "format_version": 1,
                "global_step": self.scheduler.completed_steps,
                "micro_step": 0,
                "samples_seen": self.samples_seen,
                "tokens_seen": self.tokens_seen,
                "optimizer_steps": self.scheduler.completed_steps,
                "skipped_steps": self.skipped_steps,
                "current_learning_rate": self._last_learning_rate,
                "elapsed_training_seconds": self.elapsed_training_seconds,
            }
        )

    def restore_state(
        self,
        trainer_state: TrainerState,
        data_state: DataState,
    ) -> None:
        """Restore counters and the next unread batch after other states are loaded."""
        if trainer_state.global_step != self.scheduler.completed_steps:
            raise TrainingError(
                "trainer global_step does not match scheduler completed_steps"
            )
        expected_samples = (
            trainer_state.global_step
            * self.config.batch.micro_batch_size
            * self.config.batch.gradient_accumulation_steps
        )
        expected_tokens = (
            trainer_state.global_step * self.config.batch.tokens_per_optimizer_step
        )
        if trainer_state.samples_seen != expected_samples:
            raise TrainingError("trainer samples_seen is incompatible with config")
        if trainer_state.tokens_seen != expected_tokens:
            raise TrainingError("trainer tokens_seen is incompatible with config")
        optimizer_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if optimizer_rates != {trainer_state.current_learning_rate}:
            raise TrainingError("trainer learning rate does not match optimizer state")
        self.data_iterator.restore(data_state)
        self.samples_seen = trainer_state.samples_seen
        self.tokens_seen = trainer_state.tokens_seen
        self.skipped_steps = trainer_state.skipped_steps
        self.elapsed_training_seconds = trainer_state.elapsed_training_seconds
        self._last_learning_rate = trainer_state.current_learning_rate

    def train(self, steps: int) -> TrainingResult:
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if self.scheduler.completed_steps + steps > self.config.scheduler.total_steps:
            raise TrainingError("requested steps exceed the configured schedule")

        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        metrics: list[StepMetrics] = []
        start_tokens = self.tokens_seen

        for _ in range(steps):
            learning_rate = self.scheduler.prepare_step()
            self.optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            for _ in range(self.config.batch.gradient_accumulation_steps):
                batch = self.data_iterator.next_batch().to(
                    self.device,
                    non_blocking=True,
                )
                with self._autocast():
                    output = self.model(batch, labels=batch)
                    if output.loss is None:
                        raise TrainingError("model did not return a training loss")
                    loss = output.loss
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise TrainingError("training loss is not finite")
                losses.append(loss_value)
                (loss / self.config.batch.gradient_accumulation_steps).backward()

            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.stability.max_gradient_norm,
            )
            gradient_norm = float(gradient_norm_tensor)
            if not math.isfinite(gradient_norm):
                raise TrainingError("gradient norm is not finite")
            self.optimizer.step()
            self.scheduler.step()
            self._last_learning_rate = learning_rate
            self.samples_seen += (
                self.config.batch.micro_batch_size
                * self.config.batch.gradient_accumulation_steps
            )
            self.tokens_seen += self.config.batch.tokens_per_optimizer_step
            elapsed = time.perf_counter() - started
            metrics.append(
                StepMetrics(
                    global_step=self.scheduler.completed_steps,
                    loss=sum(losses) / len(losses),
                    gradient_norm=gradient_norm,
                    learning_rate=learning_rate,
                    samples_seen=self.samples_seen,
                    tokens_seen=self.tokens_seen,
                    elapsed_seconds=elapsed,
                )
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.elapsed_training_seconds += elapsed
        processed_tokens = self.tokens_seen - start_tokens
        peak_allocated = (
            torch.cuda.max_memory_allocated(self.device) / 1024**3
            if self.device.type == "cuda"
            else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(self.device) / 1024**3
            if self.device.type == "cuda"
            else 0.0
        )
        return TrainingResult(
            trainer_state=self.trainer_state(),
            data_state=self.data_iterator.state(),
            step_metrics=tuple(metrics),
            peak_allocated_gib=peak_allocated,
            peak_reserved_gib=peak_reserved,
            tokens_per_second=processed_tokens / elapsed,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the stage-4 Atom-5M baseline training loop."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/atom-5m-baseline.yaml"),
    )
    parser.add_argument(
        "--packed-data",
        type=Path,
        default=Path("artifacts/training-data/atom-5m-wikipedia-128-v1"),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training-runs"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    config = load_training_config(args.config, project_root=root)
    model_config = load_model_config(root / config.model.config_path)
    dataset = PackedTokenDataset(root / args.packed_data)
    identity = dataset.manifest["identity"]
    expected_identity = {
        "data_version_id": config.data.data_version_id,
        "input_sha256": config.data.split_sha256,
        "tokenizer_version_id": config.data.tokenizer_version_id,
        "tokenizer_sha256": config.data.tokenizer_sha256,
        "sequence_length": config.batch.sequence_length,
        "vocab_size": model_config.tokenizer.vocab_size,
    }
    mismatches = [
        key
        for key, expected_value in expected_identity.items()
        if identity.get(key) != expected_value
    ]
    if mismatches:
        raise TrainingError(
            f"packed data does not match training config: {', '.join(mismatches)}"
        )
    if (
        dataset.manifest["formal_training_eligible"]
        != config.data.formal_training_eligible
    ):
        raise TrainingError("packed-data training eligibility does not match config")

    set_seed(config.seed)
    model = AtomLLM(model_config)
    iterator = ResumableBatchIterator(
        dataset,
        batch_size=config.batch.micro_batch_size,
        seed=config.seed,
    )
    trainer = Trainer(model, config, iterator)
    steps = args.steps if args.steps is not None else config.scheduler.total_steps
    result = trainer.train(steps)

    project_config = ProjectConfig(
        experiment_name="atom-5m-trainer-smoke",
        seed=config.seed,
        device=config.runtime.device,
        precision=config.runtime.precision,
        output_dir=args.output_dir,
    )
    run = create_run(project_config)
    shutil.copy2(args.config, run.run_dir / "training-config.yaml")
    report_path = run.reports_dir / "training-report.json"
    report_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "data_state": asdict(result.data_state),
        "final_loss": result.step_metrics[-1].loss,
        "global_step": result.trainer_state.global_step,
        "initial_loss": result.step_metrics[0].loss,
        "peak_allocated_gib": result.peak_allocated_gib,
        "peak_reserved_gib": result.peak_reserved_gib,
        "run_id": run.run_id,
        "samples_seen": result.trainer_state.samples_seen,
        "tokens_per_second": result.tokens_per_second,
        "tokens_seen": result.trainer_state.tokens_seen,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
