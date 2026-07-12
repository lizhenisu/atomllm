import json
from pathlib import Path

from atomllm.training.monitoring import TrainingMonitor
from atomllm.training.trainer import StepMetrics


def metric(step: int) -> StepMetrics:
    return StepMetrics(
        global_step=step,
        loss=3.0 / step,
        gradient_norm=0.5,
        learning_rate=1e-4,
        samples_seen=step,
        tokens_seen=step * 128,
        elapsed_seconds=float(step),
    )


def test_monitor_writes_console_jsonl_and_resumable_steps(
    tmp_path: Path, capsys
) -> None:
    with TrainingMonitor(
        tmp_path,
        total_steps=4,
        tokens_per_step=128,
        start_step=0,
        tensorboard=False,
    ) as monitor:
        first = monitor.record(metric(1))
        second = monitor.record(metric(2))

    assert first is not None and second is not None
    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert [event["global_step"] for event in events] == [1, 2]
    assert all(event["tokens_per_second"] > 0 for event in events)
    assert "[train] step=2/4" in capsys.readouterr().out

    with TrainingMonitor(
        tmp_path,
        total_steps=4,
        tokens_per_step=128,
        start_step=2,
        prior_elapsed_seconds=2.0,
        tensorboard=False,
    ) as monitor:
        monitor.record(metric(3))

    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert [event["global_step"] for event in events] == [1, 2, 3]
    assert events[-1]["elapsed_seconds"] >= 2.0


def test_monitor_creates_tensorboard_event_file(tmp_path: Path) -> None:
    with TrainingMonitor(
        tmp_path,
        total_steps=1,
        tokens_per_step=128,
        start_step=0,
        tensorboard=True,
    ) as monitor:
        monitor.record(metric(1))

    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
