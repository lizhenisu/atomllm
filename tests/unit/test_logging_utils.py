import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from atomllm.config import ProjectConfig
from atomllm.experiment import RunContext, create_run
from atomllm.logging_utils import configure_logging


@pytest.fixture
def configured_loggers() -> Iterator[list[logging.Logger]]:
    loggers: list[logging.Logger] = []
    yield loggers
    for logger in loggers:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger._atomllm_configured = False


def make_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str = "synthetic-logging",
) -> RunContext:
    monkeypatch.setattr(
        "atomllm.experiment._current_timestamp", lambda: "20260703-153000"
    )
    return create_run(
        ProjectConfig(
            experiment_name=experiment_name,
            seed=42,
            device="cuda",
            precision="bf16",
            output_dir=tmp_path / "artifacts",
        )
    )


def read_json_lines(log_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]


def track_logger(
    logger: logging.Logger, configured_loggers: list[logging.Logger]
) -> logging.Logger:
    configured_loggers.append(logger)
    return logger


def test_configure_logging_creates_jsonl_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_loggers: list[logging.Logger],
) -> None:
    run = make_run(tmp_path, monkeypatch)

    track_logger(configure_logging(run), configured_loggers)

    assert (run.logs_dir / "run.jsonl").is_file()


def test_file_log_contains_required_fields_and_unicode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_loggers: list[logging.Logger],
) -> None:
    run = make_run(tmp_path, monkeypatch)
    logger = track_logger(configure_logging(run), configured_loggers)

    logger.info("合成训练开始")
    records = read_json_lines(run.logs_dir / "run.jsonl")

    assert len(records) == 1
    assert records[0]["level"] == "INFO"
    assert records[0]["module"] == "test_logging_utils"
    assert records[0]["run_id"] == run.run_id
    assert records[0]["message"] == "合成训练开始"
    assert isinstance(records[0]["timestamp"], str)
    assert records[0]["timestamp"].endswith("Z")


def test_console_log_is_human_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_loggers: list[logging.Logger],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = make_run(tmp_path, monkeypatch)
    logger = track_logger(configure_logging(run), configured_loggers)

    logger.warning("synthetic warning")
    captured = capsys.readouterr()

    assert "WARNING" in captured.err
    assert run.run_id in captured.err
    assert "synthetic warning" in captured.err


def test_repeated_configuration_does_not_duplicate_handlers_or_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_loggers: list[logging.Logger],
) -> None:
    run = make_run(tmp_path, monkeypatch)

    first_logger = track_logger(configure_logging(run), configured_loggers)
    second_logger = configure_logging(run)
    second_logger.info("exactly once")

    assert second_logger is first_logger
    assert len(first_logger.handlers) == 2
    assert len(read_json_lines(run.logs_dir / "run.jsonl")) == 1


def test_reconfiguration_updates_log_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_loggers: list[logging.Logger],
) -> None:
    run = make_run(tmp_path, monkeypatch)
    logger = track_logger(configure_logging(run, "INFO"), configured_loggers)

    configure_logging(run, "ERROR")
    logger.warning("filtered warning")
    logger.error("visible error")
    records = read_json_lines(run.logs_dir / "run.jsonl")

    assert [record["message"] for record in records] == ["visible error"]


def test_different_runs_write_to_separate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_loggers: list[logging.Logger],
) -> None:
    first_run = make_run(tmp_path, monkeypatch, "synthetic-first")
    second_run = make_run(tmp_path, monkeypatch, "synthetic-second")
    first_logger = track_logger(configure_logging(first_run), configured_loggers)
    second_logger = track_logger(configure_logging(second_run), configured_loggers)

    first_logger.info("first run only")
    second_logger.info("second run only")
    first_records = read_json_lines(first_run.logs_dir / "run.jsonl")
    second_records = read_json_lines(second_run.logs_dir / "run.jsonl")

    assert [record["message"] for record in first_records] == ["first run only"]
    assert [record["message"] for record in second_records] == ["second run only"]


@pytest.mark.parametrize("level", ["", "verbose"])
def test_configure_logging_rejects_invalid_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    level: str,
) -> None:
    run = make_run(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="log level|unknown log level"):
        configure_logging(run, level)


def test_configure_logging_requires_existing_logs_directory(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"
    run = RunContext(
        run_id="synthetic-run",
        run_dir=missing_path,
        checkpoints_dir=missing_path / "checkpoints",
        logs_dir=missing_path / "logs",
        reports_dir=missing_path / "reports",
        config_path=missing_path / "config.yaml",
    )

    with pytest.raises(FileNotFoundError, match="run logs directory not found"):
        configure_logging(run)
