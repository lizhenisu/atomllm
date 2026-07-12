"""Run the staged Atom-50M hyperparameter search on fixed formal data windows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ContextManager

import torch
import yaml

from atomllm.experiment import set_seed
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import TrainingConfig, file_sha256, load_training_config
from atomllm.training.data import (
    ResumableShardedBatchIterator,
    ShardedTokenDataset,
)
from atomllm.training.trainer import Trainer
from atomllm.training.trainer import TrainingError


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/training/atom-50m-experiment-plan.yaml")
VARIABLES = {
    "learning_rate",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "weight_decay",
    "max_gradient_norm",
    "scheduler",
}


class SequentialSearchError(RuntimeError):
    """Raised when a staged search plan or trial result is invalid."""


class TrialDivergedError(RuntimeError):
    """Raised when a trial crosses the configured short-run divergence boundary."""


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    value: str | int | float


@dataclass(frozen=True, slots=True)
class SearchStage:
    name: str
    variable: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class SequentialSearchPlan:
    name: str
    base_config: Path
    training_data: Path
    validation_data: Path
    train_token_budget_per_trial: int
    validation_batches: int
    validation_batch_size: int
    validation_seed: int
    final_recheck_token_budget: int
    early_stop_check_interval_steps: int
    early_stop_minimum_steps: int
    early_stop_loss_ratio: float
    output_dir: Path
    stages: tuple[SearchStage, ...]


@dataclass(frozen=True, slots=True)
class SearchSettings:
    learning_rate: float
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    max_gradient_norm: float
    scheduler: str


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SequentialSearchError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SequentialSearchError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise SequentialSearchError(f"{field} must be a positive integer")
    return value


def _candidate_value(value: Any, variable: str, context: str) -> str | int | float:
    if variable in {
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "max_gradient_norm",
    }:
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise SequentialSearchError(f"{context}.value must be finite")
        result = float(value)
        if variable == "weight_decay":
            if result < 0:
                raise SequentialSearchError(f"{context}.value must be non-negative")
        elif result <= 0:
            raise SequentialSearchError(f"{context}.value must be positive")
        if variable == "warmup_ratio" and result >= 1:
            raise SequentialSearchError(f"{context}.value must be less than 1")
        return result
    if variable == "gradient_accumulation_steps":
        return _positive_int(value, f"{context}.value")
    if variable == "scheduler":
        if value not in {"cosine", "constant"}:
            raise SequentialSearchError(f"{context}.value must be cosine or constant")
        return value
    raise SequentialSearchError(f"unsupported variable: {variable}")


def load_sequential_search_plan(
    path: str | Path = DEFAULT_CONFIG,
) -> SequentialSearchPlan:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SequentialSearchError(
            f"cannot read search plan: {config_path}"
        ) from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SequentialSearchError("search plan must be a mapping")
    expected = {
        "schema_version",
        "name",
        "base_config",
        "training_data",
        "validation_data",
        "train_token_budget_per_trial",
        "validation_batches",
        "validation_batch_size",
        "validation_seed",
        "final_recheck_token_budget",
        "early_stop_check_interval_steps",
        "early_stop_minimum_steps",
        "early_stop_loss_ratio",
        "output_dir",
        "stages",
    }
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise SequentialSearchError(f"search plan missing fields: {', '.join(missing)}")
    if unknown:
        raise SequentialSearchError(
            f"search plan has unknown fields: {', '.join(unknown)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise SequentialSearchError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise SequentialSearchError("name must be a non-empty string")
    raw_stages = value["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise SequentialSearchError("stages must be a non-empty list")
    stages: list[SearchStage] = []
    seen_variables: set[str] = set()
    for stage_index, raw_stage in enumerate(raw_stages):
        context = f"stages[{stage_index}]"
        if not isinstance(raw_stage, dict) or set(raw_stage) != {
            "name",
            "variable",
            "candidates",
        }:
            raise SequentialSearchError(f"{context} fields are invalid")
        name = raw_stage["name"]
        variable = raw_stage["variable"]
        if not isinstance(name, str) or not name:
            raise SequentialSearchError(f"{context}.name must be non-empty")
        if variable not in VARIABLES:
            raise SequentialSearchError(f"{context}.variable is unsupported")
        if variable in seen_variables:
            raise SequentialSearchError(f"variable {variable} appears more than once")
        seen_variables.add(variable)
        raw_candidates = raw_stage["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise SequentialSearchError(f"{context}.candidates needs at least two")
        candidates: list[Candidate] = []
        names: set[str] = set()
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate_context = f"{context}.candidates[{candidate_index}]"
            if not isinstance(raw_candidate, dict) or set(raw_candidate) != {
                "name",
                "value",
            }:
                raise SequentialSearchError(f"{candidate_context} fields are invalid")
            candidate_name = raw_candidate["name"]
            if not isinstance(candidate_name, str) or not candidate_name:
                raise SequentialSearchError(
                    f"{candidate_context}.name must be non-empty"
                )
            if candidate_name in names:
                raise SequentialSearchError(f"{context} candidate names must be unique")
            names.add(candidate_name)
            candidates.append(
                Candidate(
                    name=candidate_name,
                    value=_candidate_value(
                        raw_candidate["value"], variable, candidate_context
                    ),
                )
            )
        stages.append(SearchStage(name, variable, tuple(candidates)))
    if seen_variables != VARIABLES:
        missing_variables = ", ".join(sorted(VARIABLES - seen_variables))
        raise SequentialSearchError(
            f"search plan is missing variables: {missing_variables}"
        )
    train_budget = _positive_int(
        value["train_token_budget_per_trial"], "train_token_budget_per_trial"
    )
    final_budget = _positive_int(
        value["final_recheck_token_budget"], "final_recheck_token_budget"
    )
    if final_budget <= train_budget:
        raise SequentialSearchError(
            "final_recheck_token_budget must exceed per-trial budget"
        )
    validation_seed = value["validation_seed"]
    if type(validation_seed) is not int or validation_seed < 0:
        raise SequentialSearchError("validation_seed must be non-negative")
    early_interval = _positive_int(
        value["early_stop_check_interval_steps"],
        "early_stop_check_interval_steps",
    )
    early_minimum = _positive_int(
        value["early_stop_minimum_steps"], "early_stop_minimum_steps"
    )
    if early_minimum < early_interval * 2:
        raise SequentialSearchError(
            "early_stop_minimum_steps must cover at least two check intervals"
        )
    early_ratio = value["early_stop_loss_ratio"]
    if type(early_ratio) not in {int, float} or not math.isfinite(float(early_ratio)):
        raise SequentialSearchError("early_stop_loss_ratio must be finite")
    early_ratio = float(early_ratio)
    if early_ratio <= 1:
        raise SequentialSearchError("early_stop_loss_ratio must be greater than 1")
    return SequentialSearchPlan(
        name=value["name"],
        base_config=_safe_path(value["base_config"], "base_config"),
        training_data=_safe_path(value["training_data"], "training_data"),
        validation_data=_safe_path(value["validation_data"], "validation_data"),
        train_token_budget_per_trial=train_budget,
        validation_batches=_positive_int(
            value["validation_batches"], "validation_batches"
        ),
        validation_batch_size=_positive_int(
            value["validation_batch_size"], "validation_batch_size"
        ),
        validation_seed=validation_seed,
        final_recheck_token_budget=final_budget,
        early_stop_check_interval_steps=early_interval,
        early_stop_minimum_steps=early_minimum,
        early_stop_loss_ratio=early_ratio,
        output_dir=_safe_path(value["output_dir"], "output_dir"),
        stages=tuple(stages),
    )


def baseline_search_settings(config: TrainingConfig) -> SearchSettings:
    return SearchSettings(
        learning_rate=config.optimizer.learning_rate,
        gradient_accumulation_steps=config.batch.gradient_accumulation_steps,
        warmup_ratio=config.scheduler.warmup_steps / config.scheduler.total_steps,
        weight_decay=config.optimizer.weight_decay,
        max_gradient_norm=config.stability.max_gradient_norm,
        scheduler=config.scheduler.name,
    )


def apply_candidate(
    settings: SearchSettings, variable: str, value: str | int | float
) -> SearchSettings:
    if variable not in VARIABLES:
        raise SequentialSearchError(f"unsupported variable: {variable}")
    return replace(settings, **{variable: value})


def _trial_config(
    base: TrainingConfig,
    settings: SearchSettings,
    *,
    token_budget: int,
    name: str,
) -> TrainingConfig:
    tokens_per_micro_batch = base.batch.sequence_length * base.batch.micro_batch_size
    tokens_per_step = tokens_per_micro_batch * settings.gradient_accumulation_steps
    if token_budget % tokens_per_step != 0:
        raise SequentialSearchError(
            f"token budget {token_budget} is not divisible by {tokens_per_step}"
        )
    total_steps = token_budget // tokens_per_step
    warmup_steps = max(1, round(total_steps * settings.warmup_ratio))
    if warmup_steps >= total_steps:
        raise SequentialSearchError("warmup consumes the complete trial")
    return replace(
        base,
        name=name,
        batch=replace(
            base.batch,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
        ),
        optimizer=replace(
            base.optimizer,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
        ),
        scheduler=replace(
            base.scheduler,
            name=settings.scheduler,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
        stability=replace(
            base.stability,
            max_gradient_norm=settings.max_gradient_norm,
        ),
    )


def _autocast(config: TrainingConfig) -> ContextManager[None]:
    if config.runtime.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_validation_loss(
    model: AtomLLM,
    dataset: ShardedTokenDataset,
    config: TrainingConfig,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> float:
    if batches * batch_size > len(dataset):
        raise SequentialSearchError("validation window exceeds validation dataset")
    model.eval()
    losses: list[float] = []
    device = torch.device(config.runtime.device)
    iterator = ResumableShardedBatchIterator(
        dataset,
        batch_size=batch_size,
        seed=seed,
    )
    for _ in range(batches):
        batch = iterator.next_batch().to(device)
        with _autocast(config):
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss).item():
            raise SequentialSearchError("validation loss is not finite")
        losses.append(float(output.loss))
    return sum(losses) / len(losses)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _run_trial(
    *,
    root: Path,
    model_config_path: Path,
    training_dataset: ShardedTokenDataset,
    validation_dataset: ShardedTokenDataset,
    config: TrainingConfig,
    validation_batches: int,
    validation_batch_size: int,
    validation_seed: int,
    early_stop_check_interval_steps: int,
    early_stop_minimum_steps: int,
    early_stop_loss_ratio: float,
) -> dict[str, Any]:
    set_seed(config.seed)
    model_config = load_model_config(model_config_path)
    model = AtomLLM(model_config)
    iterator = ResumableShardedBatchIterator(
        training_dataset,
        batch_size=config.batch.micro_batch_size,
        seed=config.seed,
    )
    trainer = Trainer(model, config, iterator)
    started = time.perf_counter()
    step_metrics = []
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    while trainer.scheduler.completed_steps < config.scheduler.total_steps:
        remaining = config.scheduler.total_steps - trainer.scheduler.completed_steps
        chunk_steps = min(early_stop_check_interval_steps, remaining)
        chunk = trainer.train(chunk_steps)
        step_metrics.extend(chunk.step_metrics)
        peak_allocated_gib = max(peak_allocated_gib, chunk.peak_allocated_gib)
        peak_reserved_gib = max(peak_reserved_gib, chunk.peak_reserved_gib)
        if len(step_metrics) >= early_stop_minimum_steps:
            baseline_loss = (
                sum(
                    metric.loss
                    for metric in step_metrics[:early_stop_check_interval_steps]
                )
                / early_stop_check_interval_steps
            )
            recent_loss = (
                sum(
                    metric.loss
                    for metric in step_metrics[-early_stop_check_interval_steps:]
                )
                / early_stop_check_interval_steps
            )
            if recent_loss > baseline_loss * early_stop_loss_ratio:
                raise TrialDivergedError(
                    f"recent loss {recent_loss:.6f} exceeds initial-window loss "
                    f"{baseline_loss:.6f} by ratio {early_stop_loss_ratio:.3f}"
                )
    training_elapsed = time.perf_counter() - started
    validation_loss = evaluate_validation_loss(
        trainer.model,
        validation_dataset,
        config,
        batches=validation_batches,
        batch_size=validation_batch_size,
        seed=validation_seed,
    )
    report = {
        "name": config.name,
        "settings": {
            "learning_rate": config.optimizer.learning_rate,
            "gradient_accumulation_steps": config.batch.gradient_accumulation_steps,
            "tokens_per_optimizer_step": config.batch.tokens_per_optimizer_step,
            "warmup_steps": config.scheduler.warmup_steps,
            "weight_decay": config.optimizer.weight_decay,
            "max_gradient_norm": config.stability.max_gradient_norm,
            "scheduler": config.scheduler.name,
        },
        "optimizer_steps": config.scheduler.total_steps,
        "tokens_seen": trainer.trainer_state().tokens_seen,
        "initial_training_loss": step_metrics[0].loss,
        "final_training_loss": step_metrics[-1].loss,
        "validation_loss": validation_loss,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "tokens_per_second": trainer.trainer_state().tokens_seen / training_elapsed,
        "training_elapsed_seconds": training_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    del trainer, iterator, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _materialize_final_config(
    *,
    root: Path,
    base_path: Path,
    base: TrainingConfig,
    settings: SearchSettings,
    token_budget: int,
    output_path: Path,
) -> None:
    final = _trial_config(
        base,
        settings,
        token_budget=token_budget,
        name="atom-50m-stage4-final-candidate",
    )
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    raw["name"] = final.name
    raw["batch"]["gradient_accumulation_steps"] = (
        final.batch.gradient_accumulation_steps
    )
    raw["optimizer"]["learning_rate"] = final.optimizer.learning_rate
    raw["optimizer"]["weight_decay"] = final.optimizer.weight_decay
    raw["scheduler"]["name"] = final.scheduler.name
    raw["scheduler"]["warmup_steps"] = final.scheduler.warmup_steps
    raw["scheduler"]["total_steps"] = final.scheduler.total_steps
    raw["stability"]["max_gradient_norm"] = final.stability.max_gradient_norm
    raw["checkpoint"]["save_every_steps"] = final.scheduler.total_steps // 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_training_config(output_path, project_root=root)


def run_sequential_search(
    plan_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan_file = root / plan_path
    plan = load_sequential_search_plan(plan_file)
    base_path = root / plan.base_config
    base = load_training_config(base_path, project_root=root)
    training_dataset = ShardedTokenDataset(
        root / plan.training_data, sequence_length=base.batch.sequence_length
    )
    validation_dataset = ShardedTokenDataset(
        root / plan.validation_data, sequence_length=base.batch.sequence_length
    )
    if training_dataset.manifest.get("split") != "train":
        raise SequentialSearchError("training_data must contain the train split")
    if validation_dataset.manifest.get("split") != "validation":
        raise SequentialSearchError("validation_data must contain the validation split")
    identity = {
        "plan_sha256": file_sha256(plan_file),
        "base_config_sha256": file_sha256(base_path),
        "training_manifest_sha256": training_dataset.manifest_sha256,
        "validation_manifest_sha256": validation_dataset.manifest_sha256,
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_dir = root / plan.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise SequentialSearchError("existing search state identity mismatch")
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "identity_sha256": identity_sha,
            "completed_stages": [],
            "settings": asdict(baseline_search_settings(base)),
        }
        _atomic_json(state_path, state)
    completed_stages = state["completed_stages"]
    settings = SearchSettings(**state["settings"])
    for stage_index in range(len(completed_stages), len(plan.stages)):
        stage = plan.stages[stage_index]
        reports: list[dict[str, Any]] = []
        for candidate in stage.candidates:
            candidate_settings = apply_candidate(
                settings, stage.variable, candidate.value
            )
            trial_name = f"{stage.name}-{candidate.name}"
            trial_config = _trial_config(
                base,
                candidate_settings,
                token_budget=plan.train_token_budget_per_trial,
                name=trial_name,
            )
            print(f"[sequential-search] start {trial_name}", flush=True)
            try:
                report = _run_trial(
                    root=root,
                    model_config_path=root / base.model.config_path,
                    training_dataset=training_dataset,
                    validation_dataset=validation_dataset,
                    config=trial_config,
                    validation_batches=plan.validation_batches,
                    validation_batch_size=plan.validation_batch_size,
                    validation_seed=plan.validation_seed,
                    early_stop_check_interval_steps=(
                        plan.early_stop_check_interval_steps
                    ),
                    early_stop_minimum_steps=plan.early_stop_minimum_steps,
                    early_stop_loss_ratio=plan.early_stop_loss_ratio,
                )
            except (TrialDivergedError, TrainingError, torch.OutOfMemoryError) as error:
                report = {
                    "name": trial_name,
                    "passed": False,
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                    "validation_loss": None,
                }
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            report["candidate"] = asdict(candidate)
            report["candidate_settings"] = asdict(candidate_settings)
            reports.append(report)
            _atomic_json(output_dir / f"{trial_name}.json", report)
            print(
                f"[sequential-search] finish {trial_name} "
                f"passed={report['passed']} "
                f"validation_loss={report['validation_loss']}",
                flush=True,
            )
        successful_reports = [report for report in reports if report["passed"]]
        if not successful_reports:
            raise SequentialSearchError(f"all trials failed in stage {stage.name}")
        winner = min(successful_reports, key=lambda report: report["validation_loss"])
        settings = SearchSettings(**winner["candidate_settings"])
        stage_report = {
            "stage": stage.name,
            "variable": stage.variable,
            "winner": winner["name"],
            "winner_validation_loss": winner["validation_loss"],
            "trials": reports,
        }
        completed_stages.append(stage_report)
        state["settings"] = asdict(settings)
        _atomic_json(state_path, state)
        _atomic_json(
            output_dir / f"stage-{stage_index + 1:02d}-{stage.name}.json", stage_report
        )
    final_config_path = output_dir / "atom-50m-final-candidate.yaml"
    _materialize_final_config(
        root=root,
        base_path=base_path,
        base=base,
        settings=settings,
        token_budget=plan.final_recheck_token_budget,
        output_path=final_config_path,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "final_settings": asdict(settings),
        "completed_stage_count": len(completed_stages),
        "final_config": str(final_config_path.relative_to(root)),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def inspect_sequential_search(
    plan_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate the staged plan and budgets without loading token shards."""
    root = Path(project_root).resolve()
    plan_file = root / plan_path
    plan = load_sequential_search_plan(plan_file)
    base = load_training_config(root / plan.base_config, project_root=root)
    micro_tokens = base.batch.sequence_length * base.batch.micro_batch_size
    accumulation_values = next(
        stage.candidates
        for stage in plan.stages
        if stage.variable == "gradient_accumulation_steps"
    )
    trial_steps: dict[str, int] = {}
    for candidate in accumulation_values:
        tokens_per_step = micro_tokens * int(candidate.value)
        for budget_name, budget in (
            ("trial", plan.train_token_budget_per_trial),
            ("final", plan.final_recheck_token_budget),
        ):
            if budget % tokens_per_step != 0:
                raise SequentialSearchError(
                    f"{budget_name} token budget is not divisible by "
                    f"candidate {candidate.name} tokens per step"
                )
        trial_steps[candidate.name] = (
            plan.train_token_budget_per_trial // tokens_per_step
        )
    return {
        "name": plan.name,
        "model": base.model.name,
        "parameter_count": base.model.expected_parameter_count,
        "stage_count": len(plan.stages),
        "trial_count": sum(len(stage.candidates) for stage in plan.stages),
        "train_token_budget_per_trial": plan.train_token_budget_per_trial,
        "final_recheck_token_budget": plan.final_recheck_token_budget,
        "early_stop": {
            "check_interval_steps": plan.early_stop_check_interval_steps,
            "minimum_steps": plan.early_stop_minimum_steps,
            "loss_ratio": plan.early_stop_loss_ratio,
        },
        "trial_steps_by_accumulation_candidate": trial_steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate staged variables and token budgets without loading data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                inspect_sequential_search(
                    args.config,
                    project_root=args.project_root,
                ),
                sort_keys=True,
            )
        )
        return 0
    result = run_sequential_search(args.config, project_root=args.project_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
