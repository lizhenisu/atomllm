import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import atomllm.training.public_token_shards as public_shards
from atomllm.data.public_tokenizer_corpus import load_config
from atomllm.training.public_token_shards import (
    COMPLETED_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    _ShardWriter,
    _padded_length,
    _sha256,
    _source_priority,
    _verify_shard,
    _write_json,
    assemble_groups,
    build_group,
    build_group_with_retries,
    tokenizer_from_gpu_selection,
)
from atomllm.training.data import ShardedTokenDataset
from atomllm.training.config import load_training_config
from atomllm.training.trainer import public_sharded_binding_mismatches


def test_padded_length_is_sequence_aligned() -> None:
    assert _padded_length(4096, 4096) == 4096
    assert _padded_length(4097, 4096) == 8192


def test_source_priority_places_curated_knowledge_before_general_web() -> None:
    sources = load_config().sources
    english = sorted(
        (source for source in sources if source.language == "en"),
        key=_source_priority,
    )

    assert [source.content_type for source in english] == [
        "encyclopedia",
        "science",
        "math",
        "general",
    ]


def test_shard_writer_adds_boundaries_and_sequence_padding(tmp_path) -> None:
    writer = _ShardWriter(
        output_dir=tmp_path,
        shard_index=0,
        sequence_length=8,
        eos_token_id=3,
        source_id="public-test",
    )
    writer.append([10, 11], b"a" * 32)

    item = writer.finish()

    assert item["content_token_count"] == 4
    assert item["padding_tokens"] == 4
    assert item["token_count"] == 8
    assert np.fromfile(tmp_path / "part-00000.bin", dtype="<u2").tolist() == [
        2,
        10,
        11,
        3,
        3,
        3,
        3,
        3,
    ]
    _verify_shard(tmp_path, item)


def test_gpu_selection_resolves_exact_selected_tokenizer(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tokenizer_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    report = {
        "training_eligible": True,
        "gpu_confirmed": True,
        "selected_tokenizer_dir": "tokenizer",
        "selected_tokenizer_sha256": _sha256(tokenizer_dir / "tokenizer.json"),
        "selected_tokenizer_manifest_sha256": _sha256(tokenizer_dir / "manifest.json"),
    }
    report_path = selection_dir / "report.json"
    _write_json(report_path, report)
    (selection_dir / COMPLETED_NAME).write_text(
        f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
    )

    selected, selection_sha = tokenizer_from_gpu_selection(
        Path("selection"), project_root=tmp_path
    )

    assert selected == tokenizer_dir
    assert selection_sha == _sha256(report_path)


def test_build_group_writes_resumable_verified_shards(
    tmp_path: Path, monkeypatch
) -> None:
    source = next(item for item in load_config().sources if item.language == "en")
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<unk>": 1,
                "<bos>": 2,
                "<eos>": 3,
                "alpha": 4,
                "beta": 5,
                "gamma": 6,
                "delta": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer_manifest = tokenizer_dir / "manifest.json"
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("test plan\n", encoding="utf-8")
    plan = SimpleNamespace(
        source_registry=Path("registry.yaml"),
        source_target_tokens={source.source_id: 20},
        shard_token_capacity=16,
        sequence_length=8,
        training_split="all-selected-documents",
        validation_status="deferred",
    )
    records = [
        {"text": text, "id": str(index)}
        for index, text in enumerate(
            (
                "alpha beta",
                "alpha gamma",
                "alpha delta",
                "beta gamma",
                "rejected record",
                "alpha beta",
                "beta delta",
            )
        )
    ]
    records[4]["id"] = "reject"

    class CheckpointableRecords:
        def __init__(self, position: int) -> None:
            self.position = position

        def __iter__(self):
            while self.position < len(records):
                record = records[self.position]
                self.position += 1
                yield record

        def state_dict(self):
            return {"position": self.position}

        def load_state_dict(self, state) -> None:
            self.position = state["position"]

    monkeypatch.setattr(public_shards, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        public_shards,
        "load_source_registry",
        lambda *_args, **_kwargs: SimpleNamespace(sources=(source,)),
    )
    monkeypatch.setattr(
        public_shards,
        "verify_tokenizer_directory",
        lambda *_args, **_kwargs: (
            tokenizer,
            {"training_eligible": True},
            tokenizer_manifest,
        ),
    )
    monkeypatch.setattr(
        public_shards,
        "_iter_source",
        lambda _source, skip: CheckpointableRecords(skip),
    )
    monkeypatch.setattr(
        public_shards,
        "_resume_source_dataset",
        lambda _source, *, records_seen, iterator_checkpoint: _resume_records(
            records_seen, iterator_checkpoint
        ),
    )
    monkeypatch.setattr(
        public_shards,
        "_accepted_text",
        lambda _source, record: (
            None if record["id"] == "reject" else (record["text"], record["id"])
        ),
    )

    def interrupted_append(self, token_ids, digest) -> None:
        interrupted_append.calls += 1
        if interrupted_append.calls == 5:
            raise RuntimeError("simulated encoder interruption")
        original_append(self, token_ids, digest)

    def _resume_records(records_seen, iterator_checkpoint):
        assert iterator_checkpoint["records_seen"] == records_seen
        dataset = CheckpointableRecords(iterator_checkpoint["base_skip_records"])
        dataset.load_state_dict(iterator_checkpoint["dataset_state"])
        return dataset

    original_append = public_shards._ShardWriter.append
    interrupted_append.calls = 0
    monkeypatch.setattr(public_shards._ShardWriter, "append", interrupted_append)

    with pytest.raises(RuntimeError, match="simulated encoder interruption"):
        build_group(
            plan_path=plan_path.relative_to(tmp_path),
            tokenizer_dir=tokenizer_dir.relative_to(tmp_path),
            output_root=Path("shards"),
            group="en",
            workers=1,
            encode_batch_size=4,
            project_root=tmp_path,
        )

    interrupted_state = json.loads(
        (tmp_path / "shards/en/state.json").read_text(encoding="utf-8")
    )
    source_checkpoint = interrupted_state["source_iterator_states"][source.source_id]
    assert source_checkpoint["records_seen"] == 4
    assert source_checkpoint["replay_through_records_seen"] == 6
    monkeypatch.setattr(public_shards._ShardWriter, "append", original_append)

    first = build_group(
        plan_path=plan_path.relative_to(tmp_path),
        tokenizer_dir=tokenizer_dir.relative_to(tmp_path),
        output_root=Path("shards"),
        group="en",
        workers=1,
        encode_batch_size=4,
        project_root=tmp_path,
    )
    second = build_group(
        plan_path=plan_path.relative_to(tmp_path),
        tokenizer_dir=tokenizer_dir.relative_to(tmp_path),
        output_root=Path("shards"),
        group="en",
        workers=1,
        encode_batch_size=4,
        project_root=tmp_path,
    )

    assert second == first
    assert first["content_token_count"] == 20
    assert first["token_count"] == 24
    assert first["document_count"] == 5
    assert first["duplicate_documents"] == 1
    assert first["rejected_documents"] == 1
    assert "excluded_validation_documents" not in first
    assert first["training_split"] == "all-selected-documents"
    assert first["validation_status"] == "deferred"
    assert first["identity"]["validation_exclusion"] is None
    assert first["identity"]["chinese_script_classifier"]["backend"] == (
        "opencc-python-reimplemented"
    )
    assert len(first["shards"]) == 2
    assert not (tmp_path / "shards/en/state.json").exists()


def test_build_group_carries_exhausted_source_shortfall_forward(
    tmp_path: Path, monkeypatch
) -> None:
    base_source = next(item for item in load_config().sources if item.language == "en")
    first_source = replace(
        base_source, source_id="first-source", content_type="encyclopedia"
    )
    second_source = replace(
        base_source, source_id="second-source", content_type="general"
    )
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<unk>": 1,
                "<bos>": 2,
                "<eos>": 3,
                "alpha": 4,
                "beta": 5,
                "gamma": 6,
                "delta": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer_manifest = tokenizer_dir / "manifest.json"
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("test plan\n", encoding="utf-8")
    plan = SimpleNamespace(
        source_registry=Path("registry.yaml"),
        source_target_tokens={"first-source": 12, "second-source": 12},
        shard_token_capacity=16,
        sequence_length=8,
        training_split="all-selected-documents",
        validation_status="deferred",
    )
    records = {
        "first-source": [{"text": "alpha beta", "id": "first-1"}],
        "second-source": [
            {"text": text, "id": f"second-{index}"}
            for index, text in enumerate(
                (
                    "alpha gamma",
                    "alpha delta",
                    "beta gamma",
                    "beta delta",
                    "gamma delta",
                )
            )
        ],
    }
    monkeypatch.setattr(public_shards, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        public_shards,
        "load_source_registry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sources=(second_source, first_source)
        ),
    )
    monkeypatch.setattr(
        public_shards,
        "verify_tokenizer_directory",
        lambda *_args, **_kwargs: (
            tokenizer,
            {"training_eligible": True},
            tokenizer_manifest,
        ),
    )
    monkeypatch.setattr(
        public_shards,
        "_iter_source",
        lambda source, skip: iter(records[source.source_id][skip:]),
    )
    monkeypatch.setattr(
        public_shards,
        "_accepted_text",
        lambda _source, record: (record["text"], record["id"]),
    )

    manifest = build_group(
        plan_path=plan_path.relative_to(tmp_path),
        tokenizer_dir=tokenizer_dir.relative_to(tmp_path),
        output_root=Path("shards"),
        group="en",
        workers=1,
        encode_batch_size=4,
        project_root=tmp_path,
    )

    assert manifest["source_order"] == ["first-source", "second-source"]
    assert manifest["source_target_tokens"] == {
        "first-source": 12,
        "second-source": 12,
    }
    assert manifest["source_content_tokens"] == {
        "first-source": 4,
        "second-source": 20,
    }
    assert manifest["source_effective_target_tokens"] == {
        "first-source": 12,
        "second-source": 20,
    }
    assert manifest["source_exhausted"] == {
        "first-source": True,
        "second-source": False,
    }
    assert manifest["content_token_count"] == 24


def test_transient_retry_reuses_process_local_verified_shards(
    tmp_path: Path, monkeypatch
) -> None:
    source = next(item for item in load_config().sources if item.language == "en")
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<unk>": 1,
                "<bos>": 2,
                "<eos>": 3,
                "alpha": 4,
                "beta": 5,
                "gamma": 6,
                "delta": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer_manifest = tokenizer_dir / "manifest.json"
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("test plan\n", encoding="utf-8")
    plan = SimpleNamespace(
        source_registry=Path("registry.yaml"),
        source_target_tokens={source.source_id: 24},
        shard_token_capacity=16,
        sequence_length=8,
        training_split="all-selected-documents",
        validation_status="deferred",
    )
    records = [
        {"text": text, "id": str(index)}
        for index, text in enumerate(
            (
                "alpha beta",
                "alpha gamma",
                "alpha delta",
                "beta gamma",
                "beta delta",
                "gamma delta",
            )
        )
    ]
    source_attempts = 0

    class FailAfterFive:
        def __init__(self, values) -> None:
            self.values = iter(values)
            self.returned = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.returned == 5:
                raise TimeoutError("simulated source timeout")
            self.returned += 1
            return next(self.values)

    def iter_source(_source, skip):
        nonlocal source_attempts
        source_attempts += 1
        values = records[skip:]
        return FailAfterFive(values) if source_attempts == 1 else iter(values)

    monkeypatch.setattr(public_shards, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        public_shards,
        "load_source_registry",
        lambda *_args, **_kwargs: SimpleNamespace(sources=(source,)),
    )
    monkeypatch.setattr(
        public_shards,
        "verify_tokenizer_directory",
        lambda *_args, **_kwargs: (
            tokenizer,
            {"training_eligible": True},
            tokenizer_manifest,
        ),
    )
    monkeypatch.setattr(public_shards, "_iter_source", iter_source)
    monkeypatch.setattr(
        public_shards,
        "_accepted_text",
        lambda _source, record: (record["text"], record["id"]),
    )
    monkeypatch.setattr(public_shards, "_reset_huggingface_http_client", lambda: None)
    monkeypatch.setattr(public_shards.time, "sleep", lambda _seconds: None)
    original_verify = public_shards._verify_shard
    verification_calls = 0

    def count_verification(*args, **kwargs) -> None:
        nonlocal verification_calls
        verification_calls += 1
        original_verify(*args, **kwargs)

    monkeypatch.setattr(public_shards, "_verify_shard", count_verification)

    manifest = build_group_with_retries(
        plan_path=plan_path.relative_to(tmp_path),
        tokenizer_dir=tokenizer_dir.relative_to(tmp_path),
        output_root=Path("shards"),
        group="en",
        workers=1,
        encode_batch_size=1,
        maximum_source_restarts=1,
        project_root=tmp_path,
    )

    assert source_attempts == 2
    assert verification_calls == 0
    assert manifest["content_token_count"] == 24
    assert manifest["document_count"] == 6


def test_assemble_groups_uses_hardlinks_and_is_training_loadable(
    tmp_path, monkeypatch
) -> None:
    groups_root = tmp_path / "groups"
    source_targets = {}
    for group_index, group in enumerate(("en", "code", "zh-Hans")):
        directory = groups_root / group
        directory.mkdir(parents=True)
        writer = _ShardWriter(
            output_dir=directory,
            shard_index=0,
            sequence_length=8,
            eos_token_id=3,
            source_id=f"source-{group}",
        )
        writer.append([10 + group_index, 20 + group_index], bytes([group_index]) * 32)
        shard = writer.finish()
        if group == "en":
            requested_targets = {"source-en-a": 3, "source-en-b": 1}
            source_order = ["source-en-a", "source-en-b"]
            effective_targets = {"source-en-a": 3, "source-en-b": 2}
            exhausted_sources = {"source-en-a": True, "source-en-b": False}
            content_tokens = {"source-en-a": 2, "source-en-b": 2}
        else:
            source_id = f"source-{group}"
            requested_targets = {source_id: 4}
            source_order = [source_id]
            effective_targets = {source_id: 4}
            exhausted_sources = {source_id: False}
            content_tokens = {source_id: 4}
        identity = {
            "format_version": FORMAT_VERSION,
            "plan_sha256": "a" * 64,
            "tokenizer_manifest_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "gpu_selection_report_sha256": "d" * 64,
            "group": group,
            "workers": 1,
            "token_dtype": "uint16-le",
            "document_boundaries": ["<bos>", "<eos>"],
            "validation_exclusion": None,
            "training_split": "all-selected-documents",
            "validation_status": "deferred",
            "synthetic_training_content": False,
            "local_text_conversion": "none",
            "local_privacy_filtering": "none",
        }
        manifest = {
            "schema_version": 1,
            "format_version": FORMAT_VERSION,
            "identity": identity,
            "vocab_size": 32000,
            "sequence_length": 8,
            "document_count": 1,
            "content_token_count": 4,
            "padding_token_count": 4,
            "token_count": 8,
            "source_target_tokens": requested_targets,
            "source_order": source_order,
            "source_effective_target_tokens": effective_targets,
            "source_exhausted": exhausted_sources,
            "source_content_tokens": content_tokens,
            "shards": [shard],
        }
        manifest_path = directory / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        (directory / COMPLETED_NAME).write_text(
            f"{_sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
        )
        source_targets.update(requested_targets)

    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("test plan\n", encoding="utf-8")
    plan_sha = _sha256(plan_path)
    for group in ("en", "code", "zh-Hans"):
        manifest_path = groups_root / group / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["identity"]["plan_sha256"] = plan_sha
        _write_json(manifest_path, manifest)
        (groups_root / group / COMPLETED_NAME).write_text(
            f"{_sha256(manifest_path)}  {MANIFEST_NAME}\n", encoding="utf-8"
        )
    plan = SimpleNamespace(
        source_target_tokens=source_targets,
        language_target_tokens={"en": 4, "code": 4, "zh-Hans": 4},
    )
    monkeypatch.setattr(public_shards, "load_plan", lambda *_args, **_kwargs: plan)

    first = assemble_groups(
        group_root=Path("groups"),
        output_dir=Path("final"),
        plan_path=Path("plan.yaml"),
        project_root=tmp_path,
    )
    second = assemble_groups(
        group_root=Path("groups"),
        output_dir=Path("final"),
        plan_path=Path("plan.yaml"),
        project_root=tmp_path,
    )

    assert second == first
    assert first["document_count"] == 3
    assert first["token_count"] == 24
    assert (groups_root / "en/part-00000.bin").stat().st_ino == (
        tmp_path / "final/part-00000.bin"
    ).stat().st_ino
    dataset = ShardedTokenDataset(tmp_path / "final", sequence_length=8)
    assert len(dataset) == 3
    assert dataset[0].tolist() == [2, 10, 20, 3, 3, 3, 3, 3]

    config = load_training_config("configs/training/atom-5m-baseline.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            data_version_id=first["dataset_id"],
            data_manifest_sha256=plan_sha,
            split_sha256=first["identity_sha256"],
            tokenizer_sha256="c" * 64,
        ),
        batch=replace(config.batch, sequence_length=8),
    )
    assert (
        public_sharded_binding_mismatches(first, config, model_vocab_size=32000) == []
    )
    drifted = replace(config, data=replace(config.data, split_sha256="d" * 64))
    assert public_sharded_binding_mismatches(
        first, drifted, model_vocab_size=32000
    ) == ["identity_sha256"]
