from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from atomllm.post_training.data_download import (
    DownloadConfig,
    PostTrainingDownloadError,
    load_download_config,
    select_datasets,
)


CONFIG_PATH = Path("configs/data/post-training-sources-v1.yaml")


@pytest.fixture
def raw_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_loads_pinned_post_training_sources() -> None:
    config = load_download_config(CONFIG_PATH)

    assert config.snapshot_id == "atom-post-training-sources-v1"
    assert len(config.datasets) == 11
    assert {stage for item in config.datasets for stage in item.stages} == set(
        range(8, 13)
    )
    assert all(len(item.revision) == 40 for item in config.datasets)
    assert sum(item.expected_bytes for item in config.datasets) == 14_054_781_884


def test_stage_selection_includes_shared_sources() -> None:
    config = load_download_config(CONFIG_PATH)

    stage_12 = select_datasets(config, {12})

    assert {item.repo_id for item in stage_12} == {
        "open-r1/OpenR1-Math-220k",
        "agentica-org/DeepScaleR-Preview-Dataset",
        "agentica-org/DeepCoder-Preview-Dataset",
    }


def test_rejects_moving_branch_revision(raw_config: dict[str, object]) -> None:
    data = deepcopy(raw_config)
    data["datasets"][0]["revision"] = "main"

    with pytest.raises(PostTrainingDownloadError, match="40-character commit"):
        DownloadConfig.from_mapping(data)


def test_rejects_unsafe_local_directory(raw_config: dict[str, object]) -> None:
    data = deepcopy(raw_config)
    data["datasets"][0]["local_dir"] = "../outside"

    with pytest.raises(PostTrainingDownloadError, match="safe relative path"):
        DownloadConfig.from_mapping(data)
