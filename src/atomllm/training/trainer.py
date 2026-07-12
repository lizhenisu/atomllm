"""AtomLLM gradient-accumulation trainer for stage-4 scale experiments."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import ContextManager
import tomllib

import torch
import yaml

from atomllm.config import ProjectConfig
from atomllm.experiment import RunContext, create_run, set_seed
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import TrainingConfig, file_sha256, load_training_config
from atomllm.training.data import (
    PackedTokenDataset,
    ResumableBatchIterator,
    ResumableShardedBatchIterator,
    ShardedTokenDataset,
)
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


@dataclass(frozen=True, slots=True)
class CheckpointEvent:
    global_step: int
    checkpoint_id: str
    manifest_sha256: str
    milestone: bool
    removed_checkpoint_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManagedTrainingResult:
    trainer_state: TrainerState
    data_state: DataState
    step_metrics: tuple[StepMetrics, ...]
    checkpoint_events: tuple[CheckpointEvent, ...]
    restored_checkpoint_id: str | None
    restored_global_step: int | None
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
        data_iterator: ResumableBatchIterator | ResumableShardedBatchIterator,
    ) -> None:
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
                    output = self.model(
                        batch,
                        labels=batch,
                        loss_chunk_size=self.config.runtime.loss_chunk_size,
                        gradient_checkpointing=(
                            self.config.runtime.gradient_checkpointing
                        ),
                        checkpoint_segment_layers=(
                            self.config.runtime.checkpoint_segment_layers
                        ),
                    )
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


def _safe_run_id(value: str) -> str:
    if not value or any(character in value for character in "/\\"):
        raise TrainingError("run_id must be a safe path segment")
    if value in {".", ".."}:
        raise TrainingError("run_id cannot be '.' or '..'")
    return value


def _project_version(project_root: Path) -> str:
    pyproject_path = project_root / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingError(
            "cannot read project version from pyproject.toml"
        ) from error
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise TrainingError("project.version is missing from pyproject.toml")
    return version


def _git_output(project_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrainingError(f"git command failed: git {' '.join(args)}") from error
    return completed.stdout.strip()


def _git_commit(project_root: Path) -> str:
    return _git_output(project_root, "rev-parse", "HEAD")


def _git_dirty(project_root: Path) -> bool:
    return bool(_git_output(project_root, "status", "--porcelain"))


def _write_config_snapshot(config: ProjectConfig, destination: Path) -> None:
    snapshot = asdict(config)
    snapshot["output_dir"] = str(config.output_dir)
    destination.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _create_named_run(config: ProjectConfig, run_id: str) -> RunContext:
    run_dir = config.output_dir / _safe_run_id(run_id)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"run directory already exists: {run_dir}") from error
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    checkpoints_dir.mkdir()
    logs_dir.mkdir()
    reports_dir.mkdir()
    config_path = run_dir / "config.yaml"
    _write_config_snapshot(config, config_path)
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        config_path=config_path,
    )


def _open_existing_run(output_dir: Path, run_id: str) -> RunContext:
    run_dir = output_dir / _safe_run_id(run_id)
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    config_path = run_dir / "config.yaml"
    for path in (run_dir, checkpoints_dir, logs_dir, reports_dir):
        if not path.is_dir():
            raise TrainingError(f"resume run directory is incomplete: {path}")
    if not config_path.is_file():
        raise TrainingError(f"resume run config snapshot is missing: {config_path}")
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        config_path=config_path,
    )


def create_or_open_run(
    project_config: ProjectConfig,
    *,
    run_id: str | None,
    resume: bool,
) -> RunContext:
    if resume:
        if run_id is None:
            raise TrainingError("--resume requires --run-id")
        return _open_existing_run(project_config.output_dir, run_id)
    if run_id is not None:
        return _create_named_run(project_config, run_id)
    return create_run(project_config)


def train_with_checkpoints(
    trainer: Trainer,
    *,
    target_steps: int,
    checkpoints_dir: Path,
    identity: object,
    save_every_steps: int,
    keep_last: int,
) -> ManagedTrainingResult:
    """Train to target_steps, saving exact-resume checkpoints on boundaries."""
    from atomllm.training.checkpoint import save_training_checkpoint

    if type(target_steps) is not int or target_steps <= 0:
        raise ValueError("target_steps must be a positive integer")
    if target_steps > trainer.config.scheduler.total_steps:
        raise TrainingError("target steps exceed the configured schedule")
    if save_every_steps <= 0:
        raise TrainingError("save_every_steps must be positive")
    current_step = trainer.trainer_state().global_step
    if target_steps < current_step:
        raise TrainingError("target steps are behind the restored checkpoint")

    metrics: list[StepMetrics] = []
    checkpoint_events: list[CheckpointEvent] = []
    peak_allocated = 0.0
    peak_reserved = 0.0
    processed_tokens = 0
    started = time.perf_counter()

    while current_step < target_steps:
        next_boundary = ((current_step // save_every_steps) + 1) * save_every_steps
        segment_target = min(target_steps, next_boundary)
        segment_steps = segment_target - current_step
        result = trainer.train(segment_steps)
        metrics.extend(result.step_metrics)
        peak_allocated = max(peak_allocated, result.peak_allocated_gib)
        peak_reserved = max(peak_reserved, result.peak_reserved_gib)
        processed_tokens += (
            segment_steps * trainer.config.batch.tokens_per_optimizer_step
        )
        current_step = result.trainer_state.global_step
        if current_step % save_every_steps == 0 or current_step == target_steps:
            milestone = current_step == target_steps
            saved = save_training_checkpoint(
                trainer,
                checkpoints_dir,
                identity,  # type: ignore[arg-type]
                keep_last=keep_last,
                milestone=milestone,
            )
            checkpoint_events.append(
                CheckpointEvent(
                    global_step=current_step,
                    checkpoint_id=saved.checkpoint_id,
                    manifest_sha256=saved.manifest_sha256,
                    milestone=milestone,
                    removed_checkpoint_ids=saved.removed_checkpoint_ids,
                )
            )

    elapsed = time.perf_counter() - started
    return ManagedTrainingResult(
        trainer_state=trainer.trainer_state(),
        data_state=trainer.data_iterator.state(),
        step_metrics=tuple(metrics),
        checkpoint_events=tuple(checkpoint_events),
        restored_checkpoint_id=None,
        restored_global_step=None,
        peak_allocated_gib=peak_allocated,
        peak_reserved_gib=peak_reserved,
        tokens_per_second=processed_tokens / elapsed if processed_tokens else 0.0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an AtomLLM model from a versioned training configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/atom-50m-baseline.yaml"),
    )
    parser.add_argument(
        "--training-data",
        "--packed-data",
        dest="training_data",
        type=Path,
        default=Path("artifacts/training-data/formal-token-shards-v2"),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Target global step. Defaults to scheduler.total_steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training-runs"),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore latest checkpoint from --run-id before training.",
    )
    parser.add_argument(
        "--checkpoint-id",
        help="Restore this checkpoint ID instead of latest when --resume is set.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Override checkpoint.save_every_steps from the training config.",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint_id is not None and not args.resume:
        raise TrainingError("--checkpoint-id requires --resume")
    root = args.project_root.resolve()
    config = load_training_config(args.config, project_root=root)
    model_config = load_model_config(root / config.model.config_path)
    data_directory = root / args.training_data
    raw_manifest = json.loads((data_directory / "manifest.json").read_text())
    if raw_manifest.get("format_version") == "document-bos-eos-sharded-v2":
        dataset = ShardedTokenDataset(
            data_directory,
            sequence_length=config.batch.sequence_length,
        )
        identity = dataset.manifest["identity"]
        expected_identity = {
            "split_manifest_sha256": config.data.split_sha256,
            "audit_manifest_sha256": config.data.data_manifest_sha256,
            "tokenizer_sha256": config.data.tokenizer_sha256,
        }
    else:
        dataset = PackedTokenDataset(data_directory)
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
    if isinstance(dataset, ShardedTokenDataset):
        iterator = ResumableShardedBatchIterator(
            dataset,
            batch_size=config.batch.micro_batch_size,
            seed=config.seed,
        )
    else:
        iterator = ResumableBatchIterator(
            dataset,
            batch_size=config.batch.micro_batch_size,
            seed=config.seed,
        )
    trainer = Trainer(model, config, iterator)
    steps = args.steps if args.steps is not None else config.scheduler.total_steps

    project_config = ProjectConfig(
        experiment_name=config.name,
        seed=config.seed,
        device=config.runtime.device,
        precision=config.runtime.precision,
        output_dir=args.output_dir,
    )
    run = create_or_open_run(
        project_config,
        run_id=args.run_id,
        resume=args.resume,
    )
    if not args.resume:
        shutil.copy2(args.config, run.run_dir / "training-config.yaml")

    from atomllm.training.checkpoint import (
        CheckpointIdentity,
        restore_training_checkpoint,
    )

    checkpoint_identity = CheckpointIdentity(
        run_id=run.run_id,
        project_version=_project_version(root),
        git_commit=_git_commit(root),
        git_dirty=_git_dirty(root),
        tokenizer_sha256=config.data.tokenizer_sha256,
        config_sha256=file_sha256(root / args.config),
    )
    restored_manifest = None
    if args.resume:
        restored_manifest = restore_training_checkpoint(
            trainer,
            run.checkpoints_dir,
            checkpoint_identity,
            selected_checkpoint_id=args.checkpoint_id,
        )

    save_every_steps = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else config.checkpoint.save_every_steps
    )
    result = train_with_checkpoints(
        trainer,
        target_steps=steps,
        checkpoints_dir=run.checkpoints_dir,
        identity=checkpoint_identity,
        save_every_steps=save_every_steps,
        keep_last=config.checkpoint.keep_last,
    )
    if restored_manifest is not None:
        result = ManagedTrainingResult(
            trainer_state=result.trainer_state,
            data_state=result.data_state,
            step_metrics=result.step_metrics,
            checkpoint_events=result.checkpoint_events,
            restored_checkpoint_id=restored_manifest["checkpoint_id"],
            restored_global_step=restored_manifest["global_step"],
            peak_allocated_gib=result.peak_allocated_gib,
            peak_reserved_gib=result.peak_reserved_gib,
            tokens_per_second=result.tokens_per_second,
        )

    report_path = (
        run.reports_dir / f"training-report-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    report_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "checkpoint_events": [asdict(event) for event in result.checkpoint_events],
        "data_state": asdict(result.data_state),
        "final_loss": result.step_metrics[-1].loss if result.step_metrics else None,
        "global_step": result.trainer_state.global_step,
        "initial_loss": result.step_metrics[0].loss if result.step_metrics else None,
        "peak_allocated_gib": result.peak_allocated_gib,
        "peak_reserved_gib": result.peak_reserved_gib,
        "run_id": run.run_id,
        "restored_checkpoint_id": result.restored_checkpoint_id,
        "restored_global_step": result.restored_global_step,
        "samples_seen": result.trainer_state.samples_seen,
        "tokens_per_second": result.tokens_per_second,
        "tokens_seen": result.trainer_state.tokens_seen,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
