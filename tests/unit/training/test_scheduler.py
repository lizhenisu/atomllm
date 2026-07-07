from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atomllm.training.config import load_training_config
from atomllm.training.scheduler import LearningRateScheduler, SchedulerError


CONFIG_PATH = Path("configs/training/atom-5m-baseline.yaml")


def make_scheduler(name: str) -> LearningRateScheduler:
    config = load_training_config(CONFIG_PATH)
    scheduler_config = replace(
        config.scheduler,
        name=name,
        warmup_steps=2,
        total_steps=6,
        minimum_learning_rate_ratio=0.1,
    )
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    return LearningRateScheduler(optimizer, scheduler_config)


def test_cosine_schedule_hits_warmup_base_and_minimum() -> None:
    scheduler = make_scheduler("cosine")

    assert scheduler.multiplier(0) == pytest.approx(0.5)
    assert scheduler.multiplier(1) == pytest.approx(1.0)
    assert scheduler.multiplier(2) == pytest.approx(1.0)
    assert scheduler.multiplier(5) == pytest.approx(0.1)


def test_constant_schedule_only_changes_during_warmup() -> None:
    scheduler = make_scheduler("constant")

    rates = []
    for _ in range(6):
        rates.append(scheduler.prepare_step())
        scheduler.step()

    assert rates == pytest.approx([0.005, 0.01, 0.01, 0.01, 0.01, 0.01])
    with pytest.raises(SchedulerError, match="already complete"):
        scheduler.prepare_step()


def test_scheduler_state_roundtrip_and_config_mismatch() -> None:
    scheduler = make_scheduler("cosine")
    scheduler.prepare_step()
    scheduler.step()
    scheduler.prepare_step()
    scheduler.step()
    state = scheduler.state_dict()

    restored = make_scheduler("cosine")
    restored.load_state_dict(state)
    assert restored.completed_steps == 2
    assert restored.prepare_step() == pytest.approx(scheduler.prepare_step())

    incompatible = make_scheduler("constant")
    with pytest.raises(SchedulerError, match="mismatch: name"):
        incompatible.load_state_dict(state)
