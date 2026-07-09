from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from atomllm.data.formal_acquisition import (
    FormalAcquisitionConfig,
    FormalAcquisitionError,
    estimate_tokens,
    load_formal_acquisition_config,
)


CONFIG_PATH = Path("configs/data/formal-v0-acquisition.yaml")


@pytest.fixture
def raw_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_loads_default_formal_acquisition_config() -> None:
    config = load_formal_acquisition_config(CONFIG_PATH)

    assert config.plan_id == "atom-formal-data-v0-acquisition"
    assert config.target_estimated_tokens == 1_000_000
    assert sum(stream.target_estimated_tokens for stream in config.streams) == 1_000_000
    assert {stream.content_type for stream in config.streams} == {
        "code",
        "encyclopedia",
        "general",
        "math",
        "science",
    }


def test_token_estimator_is_language_aware() -> None:
    assert estimate_tokens("中文测试", "zh-Hans") == 4
    assert estimate_tokens("日本語", "ja") == 3
    assert estimate_tokens("abcdefgh", "en") == 2


def test_rejects_target_sum_mismatch(raw_config: dict[str, object]) -> None:
    data = deepcopy(raw_config)
    data["streams"][0]["target_estimated_tokens"] += 1

    with pytest.raises(FormalAcquisitionError, match="stream targets"):
        FormalAcquisitionConfig.from_mapping(data)


def test_rejects_duplicate_stream_ids(raw_config: dict[str, object]) -> None:
    data = deepcopy(raw_config)
    data["streams"][1]["stream_id"] = data["streams"][0]["stream_id"]

    with pytest.raises(FormalAcquisitionError, match="stream_id values"):
        FormalAcquisitionConfig.from_mapping(data)


def test_rejects_unsafe_output_path(raw_config: dict[str, object]) -> None:
    data = deepcopy(raw_config)
    data["output_dir"] = "../outside"

    with pytest.raises(FormalAcquisitionError, match="safe relative path"):
        FormalAcquisitionConfig.from_mapping(data)


def test_missing_config_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="formal acquisition config not found"):
        load_formal_acquisition_config(tmp_path / "missing.yaml")
