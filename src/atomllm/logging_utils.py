"""Logging configuration for isolated AtomLLM experiment runs."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomllm.experiment import RunContext


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "module": record.module,
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_level(level: str) -> int:
    if not isinstance(level, str) or not level.strip():
        raise ValueError("log level must be a non-empty string")

    normalized = level.upper()
    resolved = logging.getLevelNamesMapping().get(normalized)
    if not isinstance(resolved, int):
        raise ValueError(f"unknown log level: {level}")
    return resolved


def _logger_name(run: RunContext) -> str:
    log_path = (run.logs_dir / "run.jsonl").resolve()
    path_digest = hashlib.sha256(str(log_path).encode()).hexdigest()[:12]
    return f"atomllm.run.{run.run_id}.{path_digest}"


def _build_console_handler(run_id: str, level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.addFilter(_RunIdFilter(run_id))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(module)s | %(run_id)s | %(message)s"
        )
    )
    return handler


def _build_file_handler(log_path: Path, run_id: str, level: int) -> logging.Handler:
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.addFilter(_RunIdFilter(run_id))
    handler.setFormatter(_JsonFormatter())
    return handler


def configure_logging(run: RunContext, level: str = "INFO") -> logging.Logger:
    """Configure console and JSONL logging for one experiment run."""

    resolved_level = _resolve_level(level)
    if not run.logs_dir.is_dir():
        raise FileNotFoundError(f"run logs directory not found: {run.logs_dir}")

    logger = logging.getLogger(_logger_name(run))
    logger.setLevel(resolved_level)
    logger.propagate = False

    if getattr(logger, "_atomllm_configured", False):
        for handler in logger.handlers:
            handler.setLevel(resolved_level)
        return logger

    logger.addHandler(_build_console_handler(run.run_id, resolved_level))
    logger.addHandler(
        _build_file_handler(run.logs_dir / "run.jsonl", run.run_id, resolved_level)
    )
    logger._atomllm_configured = True
    return logger
