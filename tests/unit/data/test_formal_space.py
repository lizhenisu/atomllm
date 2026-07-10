import json
from pathlib import Path

import pytest

import atomllm.data.formal_space as formal_space
from atomllm.data.formal_space import (
    FormalSpaceConfig,
    FormalSpaceError,
    acquire_formal_space,
    load_formal_space_config,
    plan_formal_space,
)


CONFIG_PATH = Path("configs/data/formal-70g-space-acquisition.yaml")


def test_loads_default_formal_70g_space_config() -> None:
    config = load_formal_space_config(CONFIG_PATH)

    assert config.plan_id == "atom-formal-data-70g-space-v1"
    assert config.target_document_bytes == 70_000_000_000
    assert sum(stream.target_document_bytes for stream in config.streams) == (
        70_000_000_000
    )
    assert {stream.stream.content_type for stream in config.streams} == {
        "code",
        "encyclopedia",
        "general",
        "math",
        "science",
    }


def test_default_70g_plan_keeps_each_source_below_twenty_percent() -> None:
    plan = plan_formal_space(CONFIG_PATH)

    assert plan["target_document_bytes"] == 70_000_000_000
    assert plan["target_document_gib"] == 65.193
    assert max(plan["source_byte_fractions"].values()) < 0.20
    assert plan["source_byte_fractions"]["fineweb2-v2-1-1"] == 0.199858
    assert plan["source_target_ceiling_bytes"] == 13_999_000_000


def test_rejects_stream_byte_target_mismatch() -> None:
    raw = json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "plan_id": "synthetic-70g",
                "formal_plan_path": "configs/data/formal-v0-sampling-plan.yaml",
                "target_document_bytes": 10,
                "output_dir": "artifacts/synthetic",
                "minimum_free_bytes": 0,
                "checkpoint_every_records": 1,
                "checkpoint_every_bytes": 1,
                "source_target_ceiling_bytes": 10,
                "exhaustion_fallback_stream_ids": ["synthetic-stream"],
                "streams": [
                    {
                        "stream_id": "synthetic-stream",
                        "loader": "hf_dataset",
                        "file_format": None,
                        "data_file": None,
                        "source_id": "wikipedia-20231101",
                        "dataset": "synthetic/dataset",
                        "config_name": "default",
                        "split": "train",
                        "revision": "0" * 40,
                        "text_field": "text",
                        "id_field": "id",
                        "title_field": None,
                        "url_field": None,
                        "language": "en",
                        "content_type": "general",
                        "initial_skip_records": 0,
                        "target_document_bytes": 9,
                    }
                ],
            }
        )
    )

    with pytest.raises(FormalSpaceError, match="stream targets"):
        FormalSpaceConfig.from_mapping(raw)


def test_acquire_formal_space_writes_manifest_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "space.yaml"
    config_path.write_text(
        """
schema_version: 1
plan_id: synthetic-space
formal_plan_path: configs/data/formal-v0-sampling-plan.yaml
target_document_bytes: 800
output_dir: out
minimum_free_bytes: 0
checkpoint_every_records: 1
checkpoint_every_bytes: 1
source_target_ceiling_bytes: 800
exhaustion_fallback_stream_ids: [synthetic-en]
streams:
  - stream_id: synthetic-en
    loader: hf_dataset
    file_format: null
    data_file: null
    source_id: fineweb-2025-07
    dataset: synthetic/dataset
    config_name: default
    split: train
    revision: "0000000000000000000000000000000000000000"
    text_field: text
    id_field: id
    title_field: null
    url_field: null
    language: en
    content_type: general
    initial_skip_records: 0
    target_document_bytes: 800
""",
        encoding="utf-8",
    )

    def fake_validate(config, project_root):
        return {
            "formal_data_plan_id": "synthetic-plan",
            "mixture_plan_id": "synthetic-mixture",
            "max_source_fraction": 1.0,
            "source_byte_targets": {"fineweb-2025-07": 800},
            "source_byte_fractions": {"fineweb-2025-07": 1.0},
        }

    def fake_iter(stream):
        for index in range(10):
            yield {"id": str(index), "text": f"synthetic document {index} " * 30}

    monkeypatch.setattr(formal_space, "_validate_plan_and_mixture", fake_validate)
    monkeypatch.setattr(formal_space, "_iter_huggingface", fake_iter)
    monkeypatch.setattr(
        formal_space,
        "_validate_completed_distribution",
        lambda state, config, validation: {"language_priority_passed": True},
    )

    manifest = acquire_formal_space(
        config_path,
        project_root=tmp_path,
        require_disk_space=False,
    )
    repeated = acquire_formal_space(
        config_path,
        project_root=tmp_path,
        require_disk_space=False,
    )
    progress = capsys.readouterr().err

    assert repeated == manifest
    assert manifest["formal_training_eligible"] is True
    assert manifest["actual_document_bytes"] >= 800
    assert manifest["record_count"] > 0
    assert (tmp_path / "out" / "documents.jsonl").is_file()
    assert (tmp_path / "out" / "state.json").is_file()
    assert "[formal-space] start" in progress
    assert "[formal-space] checkpoint" in progress
    assert "[formal-space] completed manifest=" in progress
    assert "[formal-space] already completed:" in progress


def test_exhausted_stream_redistributes_to_configured_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "space.yaml"
    config_path.write_text(
        """
schema_version: 1
plan_id: synthetic-redistribution
formal_plan_path: configs/data/formal-v0-sampling-plan.yaml
target_document_bytes: 1000
output_dir: out
minimum_free_bytes: 0
checkpoint_every_records: 1
checkpoint_every_bytes: 1
source_target_ceiling_bytes: 800
exhaustion_fallback_stream_ids: [synthetic-fallback]
streams:
  - stream_id: synthetic-short
    loader: hf_dataset
    file_format: null
    data_file: null
    source_id: wikipedia-20231101
    dataset: synthetic/dataset
    config_name: default
    split: train
    revision: "0000000000000000000000000000000000000000"
    text_field: text
    id_field: id
    title_field: null
    url_field: null
    language: en
    content_type: general
    initial_skip_records: 0
    target_document_bytes: 800
  - stream_id: synthetic-fallback
    loader: hf_dataset
    file_format: null
    data_file: null
    source_id: fineweb-2025-07
    dataset: synthetic/dataset
    config_name: default
    split: train
    revision: "0000000000000000000000000000000000000000"
    text_field: text
    id_field: id
    title_field: null
    url_field: null
    language: en
    content_type: general
    initial_skip_records: 0
    target_document_bytes: 200
""",
        encoding="utf-8",
    )

    def fake_validate(config, project_root):
        return {
            "formal_data_plan_id": "synthetic-plan",
            "mixture_plan_id": "synthetic-mixture",
            "max_source_fraction": 1.0,
            "source_byte_targets": {},
            "source_byte_fractions": {},
        }

    def fake_iter(stream):
        if stream.stream_id == "synthetic-short":
            yield {"id": "short", "text": "short stream"}
            return
        for index in range(10):
            yield {"id": str(index), "text": "fallback document " * 30}

    monkeypatch.setattr(formal_space, "_validate_plan_and_mixture", fake_validate)
    monkeypatch.setattr(formal_space, "_iter_huggingface", fake_iter)
    monkeypatch.setattr(
        formal_space,
        "_validate_completed_distribution",
        lambda state, config, validation: {"language_priority_passed": True},
    )

    manifest = acquire_formal_space(
        config_path,
        project_root=tmp_path,
        require_disk_space=False,
    )
    state = json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))

    assert (
        manifest["exhausted_streams"]["synthetic-short"]["actual_document_bytes"] < 800
    )
    assert manifest["redistributions"][0]["from_stream_id"] == "synthetic-short"
    assert state["stream_effective_target_bytes"]["synthetic-short"] < 800
    assert state["stream_effective_target_bytes"]["synthetic-fallback"] > 200
    assert (
        "[formal-space] stream-exhausted stream=synthetic-short"
        in capsys.readouterr().err
    )


def test_existing_state_is_migrated_for_dynamic_redistribution(tmp_path: Path) -> None:
    config = load_formal_space_config(CONFIG_PATH)
    validation = {"formal_data_plan_id": "synthetic-formal-plan"}
    state = formal_space._initial_state(config, validation)
    del state["stream_effective_target_bytes"]
    del state["exhausted_streams"]
    del state["redistributions"]
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

    migrated = formal_space._load_or_create_state(tmp_path, config, validation)

    assert migrated["stream_effective_target_bytes"] == {
        item.stream.stream_id: item.target_document_bytes for item in config.streams
    }
    assert migrated["exhausted_streams"] == {}
    assert migrated["redistributions"] == []


def test_rebase_effective_targets_adds_new_stream_without_rewriting_state(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "space.yaml"
    config_path.write_text(
        """
schema_version: 1
plan_id: synthetic-rebase
formal_plan_path: configs/data/formal-v0-sampling-plan.yaml
target_document_bytes: 100
output_dir: out
minimum_free_bytes: 0
checkpoint_every_records: 1
checkpoint_every_bytes: 1
source_target_ceiling_bytes: 100
exhaustion_fallback_stream_ids: [second]
streams:
  - stream_id: first
    loader: hf_dataset
    file_format: null
    data_file: null
    source_id: fineweb-2025-07
    dataset: synthetic/dataset
    config_name: default
    split: train
    revision: "0000000000000000000000000000000000000000"
    text_field: text
    id_field: id
    title_field: null
    url_field: null
    language: en
    content_type: general
    initial_skip_records: 0
    target_document_bytes: 60
  - stream_id: second
    loader: hf_dataset
    file_format: null
    data_file: null
    source_id: wikipedia-20231101
    dataset: synthetic/dataset
    config_name: default
    split: train
    revision: "0000000000000000000000000000000000000000"
    text_field: text
    id_field: id
    title_field: null
    url_field: null
    language: en
    content_type: general
    initial_skip_records: 0
    target_document_bytes: 40
""",
        encoding="utf-8",
    )
    config = load_formal_space_config(config_path)
    validation = {"formal_data_plan_id": "synthetic-formal-plan"}
    state = formal_space._initial_state(config, validation)
    state["stream_positions"] = {"first": 2}
    state["stream_document_bytes"] = {"first": 60}
    state["stream_effective_target_bytes"] = {"first": 100}
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

    rebased = formal_space._load_or_create_state(
        tmp_path,
        config,
        validation,
        rebase_effective_targets=True,
    )

    assert rebased["stream_document_bytes"] == {"first": 60, "second": 0}
    assert rebased["stream_positions"] == {"first": 2, "second": 0}
    assert rebased["stream_effective_target_bytes"] == {"first": 60, "second": 40}
