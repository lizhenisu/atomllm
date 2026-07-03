"""Project configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


VALID_PRECISIONS = frozenset({"bf16", "fp16", "fp32"})


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or fails validation."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated top-level settings shared by AtomLLM commands."""

    experiment_name: str
    seed: int
    device: str
    precision: str
    output_dir: Path

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ConfigError("experiment_name must be a non-empty string")
        if type(self.seed) is not int:
            raise ConfigError("seed must be an integer")
        if self.seed < 0:
            raise ConfigError("seed must be greater than or equal to 0")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ConfigError("device must be a non-empty string")
        if self.precision not in VALID_PRECISIONS:
            choices = ", ".join(sorted(VALID_PRECISIONS))
            raise ConfigError(f"precision must be one of: {choices}")
        if not isinstance(self.output_dir, Path):
            raise ConfigError("output_dir must be a path")


def _validate_keys(data: dict[str, Any]) -> None:
    expected = {field.name for field in fields(ProjectConfig)}
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)

    if missing:
        raise ConfigError(f"missing required fields: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"unknown fields: {', '.join(unknown)}")


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project configuration from a YAML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_data, dict):
        raise ConfigError("configuration root must be a mapping")
    if not all(isinstance(key, str) for key in raw_data):
        raise ConfigError("configuration field names must be strings")

    _validate_keys(raw_data)
    data = dict(raw_data)

    output_dir = data["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("output_dir must be a non-empty string")
    data["output_dir"] = Path(output_dir)

    try:
        return ProjectConfig(**data)
    except TypeError as error:
        raise ConfigError(f"invalid configuration values: {error}") from error
