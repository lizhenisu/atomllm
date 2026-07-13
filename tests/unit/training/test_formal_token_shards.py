import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.training.config import file_sha256
import atomllm.training.data as data_module
from atomllm.training.data import (
    ResumableShardedBatchIterator,
    ShardedTokenDataset,
    TrainingDataError,
)
from atomllm.training.formal_token_shards import (
    build_formal_token_shards,
    load_formal_token_shard_config,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_formal_fixture(tmp_path: Path) -> Path:
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "<": 6,
        "unk": 7,
        ">": 8,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
    tokenizer_path = tmp_path / "tokenizer/tokenizer.json"
    tokenizer_path.parent.mkdir()
    tokenizer.save(str(tokenizer_path))
    tokenizer_sha = file_sha256(tokenizer_path)

    split_dir = tmp_path / "split"
    shard_dir = split_dir / "train/shards"
    shard_dir.mkdir(parents=True)
    shard_metadata = []
    for shard_index in range(2):
        shard_path = shard_dir / f"part-{shard_index:05d}.jsonl"
        lines = []
        for document_index in range(2):
            record_id = f"{shard_index}-{document_index}"
            document = CanonicalDocument(
                schema_version=1,
                document_id=make_document_id("synthetic", record_id),
                source_id="synthetic",
                source_record_id=record_id,
                text="hello world",
                language="en",
                content_type="general",
                privacy_warnings=(),
                quality_warnings=(),
                metadata={},
            )
            lines.append(document.to_json_line())
        shard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        shard_metadata.append(
            {
                "name": shard_path.name,
                "record_count": 2,
                "sha256": file_sha256(shard_path),
            }
        )
    split_manifest_path = split_dir / "manifest.json"
    _write_json(split_manifest_path, {"shards": {"train": shard_metadata}})
    split_sha = file_sha256(split_manifest_path)

    audit_path = tmp_path / "audit/manifest.json"
    _write_json(
        audit_path,
        {"training_eligible": True, "provenance": {"split": split_sha}},
    )
    version_path = tmp_path / "tokenizer-version/manifest.json"
    _write_json(
        version_path,
        {
            "formal_pretraining_eligible": True,
            "tokenizer_version_id": "formal-tokenizer-test-v1",
            "contract": {"vocab_size": len(vocabulary)},
            "lineage": {"tokenizer": {"tokenizer_sha256": tokenizer_sha}},
        },
    )
    config = {
        "schema_version": 1,
        "name": "formal-test-v1",
        "split_dir": "split",
        "split_manifest_sha256": split_sha,
        "audit_manifest": "audit/manifest.json",
        "audit_manifest_sha256": file_sha256(audit_path),
        "tokenizer_version_manifest": "tokenizer-version/manifest.json",
        "tokenizer_version_manifest_sha256": file_sha256(version_path),
        "tokenizer_path": "tokenizer/tokenizer.json",
        "tokenizer_sha256": tokenizer_sha,
        "output_dir": "output",
        "token_dtype": "uint16-le",
        "max_rss_gib": 9.0,
        "progress_interval_seconds": 60,
        "workers": 1,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_formal_token_shards_resume_and_preserve_document_index(tmp_path: Path) -> None:
    config_path = _prepare_formal_fixture(tmp_path)

    partial = build_formal_token_shards(
        config_path.name,
        project_root=tmp_path,
        max_input_shards=1,
    )
    assert len(partial["shards"]) == 1
    assert (tmp_path / "output/state.json").is_file()
    assert not (tmp_path / "output/COMPLETED").exists()

    manifest = build_formal_token_shards(config_path.name, project_root=tmp_path)
    assert manifest["formal_training_eligible"] is True
    assert manifest["document_count"] == 4
    assert manifest["token_count"] == 16
    assert len(manifest["shards"]) == 2
    assert not (tmp_path / "output/state.json").exists()

    first = manifest["shards"][0]
    tokens = np.fromfile(tmp_path / "output" / first["token_file"]["name"], dtype="<u2")
    index = np.fromfile(
        tmp_path / "output" / first["index_file"]["name"], dtype="<u8"
    ).reshape(-1, 2)
    assert tokens.tolist() == [2, 4, 5, 3, 2, 4, 5, 3]
    assert index.tolist() == [[0, 4], [4, 4]]

    assert (
        build_formal_token_shards(config_path.name, project_root=tmp_path) == manifest
    )


def test_sharded_cursor_is_deterministic_across_resume_and_epoch(
    tmp_path: Path,
) -> None:
    config_path = _prepare_formal_fixture(tmp_path)
    build_formal_token_shards(config_path.name, project_root=tmp_path)
    dataset = ShardedTokenDataset(tmp_path / "output", sequence_length=4)
    uninterrupted = ResumableShardedBatchIterator(dataset, batch_size=2, seed=42)

    first = uninterrupted.next_batch()
    state = uninterrupted.state()
    expected_next = uninterrupted.next_batch()
    expected_next_epoch = uninterrupted.next_batch()

    resumed = ResumableShardedBatchIterator(dataset, batch_size=2, seed=42)
    resumed.restore(state)
    assert np.array_equal(resumed.next_batch().numpy(), expected_next.numpy())
    assert np.array_equal(resumed.next_batch().numpy(), expected_next_epoch.numpy())
    assert first.shape == (2, 4)


def test_verified_manifest_skips_rehashing_shard_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_formal_fixture(tmp_path)
    manifest = build_formal_token_shards(config_path.name, project_root=tmp_path)
    output_dir = tmp_path / "output"
    manifest_sha256 = hashlib.sha256(
        (output_dir / "manifest.json").read_bytes()
    ).hexdigest()

    def unexpected_verification(directory: Path) -> dict:
        raise AssertionError(f"unexpected full verification: {directory}")

    monkeypatch.setattr(
        data_module,
        "verify_formal_token_shards",
        unexpected_verification,
    )
    dataset = ShardedTokenDataset(
        output_dir,
        sequence_length=4,
        verified_manifest=manifest,
        manifest_sha256=manifest_sha256,
    )

    assert len(dataset) == 4
    assert dataset.manifest_sha256 == manifest_sha256

    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="does not match"):
        ShardedTokenDataset(
            output_dir,
            sequence_length=4,
            verified_manifest=manifest,
            manifest_sha256=manifest_sha256,
        )


def test_train_is_default_and_validation_can_be_selected(tmp_path: Path) -> None:
    config_path = _prepare_formal_fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "input_split" not in raw

    raw["input_split"] = "validation"
    validation_path = tmp_path / "validation.yaml"
    validation_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_formal_token_shard_config(config_path).input_split == "train"
    assert load_formal_token_shard_config(validation_path).input_split == "validation"


def test_raw_special_token_literal_is_encoded_as_text(tmp_path: Path) -> None:
    config_path = _prepare_formal_fixture(tmp_path)
    config = load_formal_token_shard_config(config_path)
    tokenizer = Tokenizer.from_file(str(tmp_path / config.tokenizer_path))

    assert 1 in tokenizer.encode("hello <unk>", add_special_tokens=False).ids
    tokenizer.encode_special_tokens = True
    assert tokenizer.encode("hello <unk>", add_special_tokens=False).ids == [4, 6, 7, 8]


def test_two_worker_encoding_commits_results_in_source_order(tmp_path: Path) -> None:
    config_path = _prepare_formal_fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["workers"] = 2
    raw["output_dir"] = "parallel-output"
    parallel_path = tmp_path / "parallel.yaml"
    parallel_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    manifest = build_formal_token_shards(parallel_path.name, project_root=tmp_path)

    assert [item["source_name"] for item in manifest["shards"]] == [
        "part-00000.jsonl",
        "part-00001.jsonl",
    ]
    assert manifest["workers"] == 2
