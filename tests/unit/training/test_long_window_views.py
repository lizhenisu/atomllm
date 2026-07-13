import json
from pathlib import Path

import pytest
import yaml

from atomllm.training.data import LongWindowDataset, ResumableShardedBatchIterator
from atomllm.training.formal_token_shards import build_formal_token_shards
from atomllm.training.long_window_views import (
    LongWindowViewError,
    build_long_window_view,
    load_long_window_view_config,
    verify_long_window_view,
)
from test_formal_token_shards import _prepare_formal_fixture


def _view_config(tmp_path: Path) -> Path:
    config = {
        "schema_version": 1,
        "name": "synthetic-long-v1",
        "stage": "B",
        "source_dir": "output",
        "output_dir": "long-view",
        "window_length": 4,
        "stride": 4,
        "expected_candidate_count": 4,
        "selection_count": 4,
    }
    path = tmp_path / "long-view.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_builds_deterministic_document_internal_view(tmp_path: Path) -> None:
    formal_config = _prepare_formal_fixture(tmp_path)
    build_formal_token_shards(formal_config.name, project_root=tmp_path)
    config = load_long_window_view_config(_view_config(tmp_path))

    first_path = build_long_window_view(config, project_root=tmp_path)
    first_bytes = first_path.read_bytes()
    second_path = build_long_window_view(config, project_root=tmp_path)

    assert second_path.read_bytes() == first_bytes
    manifest = verify_long_window_view(tmp_path / "long-view")
    assert manifest["candidate_count"] == 4
    assert manifest["selected_tokens"] == 16
    assert all(
        item["window_end"] <= item["document_token_count"]
        for item in manifest["windows"]
    )

    dataset = LongWindowDataset(tmp_path / "long-view")
    iterator = ResumableShardedBatchIterator(dataset, batch_size=2, seed=42)
    batch = iterator.next_batch()
    state = iterator.state()
    resumed = ResumableShardedBatchIterator(dataset, batch_size=2, seed=42)
    resumed.restore(state)
    assert batch.shape == (2, 4)
    assert resumed.next_batch().shape == (2, 4)


def test_rejects_source_lineage_drift(tmp_path: Path) -> None:
    formal_config = _prepare_formal_fixture(tmp_path)
    build_formal_token_shards(formal_config.name, project_root=tmp_path)
    config = load_long_window_view_config(_view_config(tmp_path))
    build_long_window_view(config, project_root=tmp_path)
    source_manifest = tmp_path / "output/manifest.json"
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    raw["dataset_id"] = "drifted"
    source_manifest.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(LongWindowViewError, match="SHA-256"):
        verify_long_window_view(tmp_path / "long-view")
