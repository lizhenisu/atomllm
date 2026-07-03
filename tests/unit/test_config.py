from pathlib import Path

import pytest

from atomllm.config import ConfigError, ProjectConfig, load_config


VALID_CONFIG = """\
experiment_name: synthetic-smoke
seed: 42
device: cuda
precision: bf16
output_dir: artifacts
"""


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_loads_valid_config(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config == ProjectConfig(
        experiment_name="synthetic-smoke",
        seed=42,
        device="cuda",
        precision="bf16",
        output_dir=Path("artifacts"),
    )


def test_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        load_config(missing_path)


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("precision: bf16\n", "")

    with pytest.raises(ConfigError, match="missing required fields: precision"):
        load_config(write_config(tmp_path, content))


def test_rejects_unknown_field(tmp_path: Path) -> None:
    content = f"{VALID_CONFIG}unexpected: value\n"

    with pytest.raises(ConfigError, match="unknown fields: unexpected"):
        load_config(write_config(tmp_path, content))


def test_rejects_negative_seed(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("seed: 42", "seed: -1")

    with pytest.raises(ConfigError, match="seed must be greater than or equal to 0"):
        load_config(write_config(tmp_path, content))


def test_rejects_non_integer_seed(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("seed: 42", "seed: synthetic")

    with pytest.raises(ConfigError, match="seed must be an integer"):
        load_config(write_config(tmp_path, content))


def test_rejects_invalid_precision(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("precision: bf16", "precision: int4")

    with pytest.raises(ConfigError, match="precision must be one of"):
        load_config(write_config(tmp_path, content))


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write_config(tmp_path, "experiment_name: [unterminated"))


@pytest.mark.parametrize("content", ["", "- not\n- a\n- mapping\n"])
def test_rejects_non_mapping_root(tmp_path: Path, content: str) -> None:
    with pytest.raises(ConfigError, match="configuration root must be a mapping"):
        load_config(write_config(tmp_path, content))
