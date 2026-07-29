from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from itertools import islice

import pytest
import yaml
from datasets import Dataset

import atomllm.data.public_tokenizer_corpus as public_corpus
from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.data.public_tokenizer_corpus import (
    PublicTokenizerCorpusError,
    _accepted_text,
    _accepted_text_worker,
    _cached_source_files,
    _content_bound_source_record_id,
    _dataset_state_cursor,
    _initialize_acceptance_worker,
    _is_transient_source_error,
    _rebuild_fingerprints,
    _resume_source_dataset,
    _usable_iterator_checkpoint,
    build_with_retries,
    load_config,
)


CONFIG = Path("configs/data/public-tokenizer-corpus-en-zh-v1.yaml")


def test_content_bound_source_record_id_disambiguates_reused_upstream_id() -> None:
    first = _content_bound_source_record_id("shared", bytes.fromhex("11" * 32))
    second = _content_bound_source_record_id("shared", bytes.fromhex("22" * 32))

    assert first != second
    assert _content_bound_source_record_id(first, bytes.fromhex("11" * 32)) == first


def test_public_tokenizer_config_enforces_fifty_ten_forty_mix() -> None:
    config = load_config(CONFIG)
    targets = {
        language: sum(
            source.target_text_bytes
            for source in config.sources
            if source.language == language
        )
        for language in {"en", "zh-Hans", "code"}
    }

    assert targets == {
        "en": 12_000_000_000,
        "zh-Hans": 9_600_000_000,
        "code": 2_400_000_000,
    }
    assert sum(targets.values()) == 24_000_000_000
    assert all(source.revision not in {"main", "master"} for source in config.sources)
    cci3 = next(source for source in config.sources if source.dataset == "BAAI/CCI3-HQ")
    assert cci3.minimum_score == 4.0
    industry = next(
        source
        for source in config.sources
        if source.source_id == "industry-corpus2-zh-high"
    )
    assert industry.data_files_pattern == "*/chinese/high/rank_*.parquet"
    assert industry.minimum_score == 4.0
    assert not hasattr(cci3, "reject_privacy_warnings")


def test_allows_declared_upstream_t2s_but_rejects_local_conversion(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["sources"][0]["text_conversion"] = "upstream-t2s"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).sources[0].language == "zh-Hans"

    raw["sources"][0]["text_conversion"] = "local-t2s"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PublicTokenizerCorpusError, match="local conversion"):
        load_config(path)

    raw["sources"][0]["text_conversion"] = "none"
    raw["sources"][0]["synthetic_content"] = True
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PublicTokenizerCorpusError, match="synthetic"):
        load_config(path)


def test_native_simplified_filter_does_not_convert_text() -> None:
    source = load_config(CONFIG).sources[0]
    simplified = "人工智能系统需要可靠的数据和严格的质量控制。" * 20
    traditional = "人工智慧系統需要可靠的資料與嚴格的品質控制。" * 20

    accepted = _accepted_text(source, {"id": "s", "text": simplified, "score": 4})

    assert accepted == (simplified, "s")
    assert (
        _accepted_text(source, {"id": "low", "text": simplified, "score": 3.99}) is None
    )
    assert _accepted_text(source, {"id": "t", "text": traditional, "score": 4}) is None


def test_parallel_classification_matches_locked_single_process_result() -> None:
    source = load_config(CONFIG).sources[0]
    records = [
        {"id": "s", "text": "人工智能需要高质量数据。" * 30, "score": 4},
        {"id": "t", "text": "人工智慧需要高品質資料。" * 30, "score": 4},
    ]

    expected = [_accepted_text(source, record) for record in records]
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
        initializer=_initialize_acceptance_worker,
        initargs=(source,),
    ) as executor:
        actual = list(executor.map(_accepted_text_worker, records))

    assert actual == expected


def test_public_contact_information_is_not_used_as_a_rejection_rule() -> None:
    source = load_config(CONFIG).sources[0]
    text = "公开论文作者邮箱 author@example.org 与联系电话 13800138000。" * 20

    assert _accepted_text(source, {"id": "public", "text": text, "score": 4}) == (
        text,
        "public",
    )


def test_source_cache_requires_matching_revision_and_sha256(
    tmp_path: Path, monkeypatch
) -> None:
    source = load_config(CONFIG).sources[3]
    relative = Path("sample/100BT/000_00004.parquet")
    repository = tmp_path / source.dataset / source.revision
    local = repository / relative
    local.parent.mkdir(parents=True)
    local.write_bytes(b"verified parquet bytes")
    metadata = (
        repository
        / ".cache/huggingface/download"
        / relative.parent
        / f"{relative.name}.metadata"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"{source.revision}\n{public_corpus._sha256(local)}\n0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(public_corpus.SOURCE_CACHE_ENV, str(tmp_path))
    remote = f"hf://datasets/{source.dataset}@{source.revision}/{relative.as_posix()}"

    files, replacements = _cached_source_files(source, [remote])

    assert files == [str(local.resolve())]
    assert replacements == 1
    local.write_bytes(b"corrupt")
    with pytest.raises(PublicTokenizerCorpusError, match="SHA-256 mismatch"):
        _cached_source_files(source, [remote])


@pytest.mark.parametrize("existing_part", ["local", "metadata"])
def test_source_cache_in_progress_entry_falls_back_to_remote(
    tmp_path: Path, monkeypatch, existing_part: str
) -> None:
    source = load_config(CONFIG).sources[3]
    relative = Path("sample/100BT/000_00004.parquet")
    repository = tmp_path / source.dataset / source.revision
    local = repository / relative
    metadata = (
        repository
        / ".cache/huggingface/download"
        / relative.parent
        / f"{relative.name}.metadata"
    )
    if existing_part == "local":
        local.parent.mkdir(parents=True)
        local.write_bytes(b"download not committed")
    else:
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"{source.revision}\n{'0' * 64}\n0\n", encoding="utf-8")
    monkeypatch.setenv(public_corpus.SOURCE_CACHE_ENV, str(tmp_path))
    remote = f"hf://datasets/{source.dataset}@{source.revision}/{relative.as_posix()}"

    files, replacements = _cached_source_files(source, [remote])

    assert files == [remote]
    assert replacements == 0


def test_source_resume_prefers_exact_iterator_checkpoint(monkeypatch) -> None:
    source = load_config(CONFIG).sources[0]

    class FakeDataset:
        def __init__(self) -> None:
            self.loaded = None
            self.skipped = None

        def load_state_dict(self, value) -> None:
            self.loaded = value

        def skip(self, count):
            self.skipped = count
            return self

    dataset = FakeDataset()
    monkeypatch.setattr(
        "atomllm.data.public_tokenizer_corpus._load_source_dataset",
        lambda _source: dataset,
    )

    resumed = _resume_source_dataset(
        source,
        records_seen=123,
        iterator_checkpoint={
            "records_seen": 123,
            "base_skip_records": 100,
            "dataset_state": {"offset": 23},
        },
    )

    assert resumed is dataset
    assert dataset.skipped == 100
    assert dataset.loaded == {"offset": 23}


def test_source_resume_rejects_mismatched_iterator_checkpoint(monkeypatch) -> None:
    source = load_config(CONFIG).sources[0]
    monkeypatch.setattr(
        "atomllm.data.public_tokenizer_corpus._load_source_dataset",
        lambda _source: object(),
    )

    with pytest.raises(PublicTokenizerCorpusError, match="checkpoint mismatch"):
        _resume_source_dataset(
            source,
            records_seen=123,
            iterator_checkpoint={"records_seen": 122, "dataset_state": {}},
        )


def test_resumed_dataset_state_continues_advancing(monkeypatch) -> None:
    source = load_config(CONFIG).sources[0]

    def local_dataset():
        return Dataset.from_dict({"value": list(range(30))}).to_iterable_dataset(
            num_shards=3
        )

    initial = local_dataset().skip(5)
    assert [item["value"] for item in islice(iter(initial), 7)] == list(range(5, 12))
    checkpoint_state = initial.state_dict()
    checkpoint_cursor = _dataset_state_cursor(checkpoint_state)
    monkeypatch.setattr(
        "atomllm.data.public_tokenizer_corpus._load_source_dataset",
        lambda _source: local_dataset(),
    )

    resumed = _resume_source_dataset(
        source,
        records_seen=12,
        iterator_checkpoint={
            "records_seen": 12,
            "base_skip_records": 5,
            "dataset_state": checkpoint_state,
        },
    )

    assert [item["value"] for item in islice(iter(resumed), 4)] == list(range(12, 16))
    assert _dataset_state_cursor(resumed.state_dict()) > checkpoint_cursor


def test_iterator_checkpoint_rejects_a_cursor_that_reset_to_zero() -> None:
    reset = {
        "records_seen": 2_660_334,
        "base_skip_records": 1_887_214,
        "dataset_state": {
            "examples_iterable": {
                "skipped": 1_887_214,
                "shard_idx": 0,
                "shard_example_idx": 0,
                "batch_idx": 0,
            }
        },
    }
    advanced = {
        **reset,
        "dataset_state_records_seen": 2_600_000,
        "dataset_state": {
            "examples_iterable": {
                "shard_idx": 13,
                "shard_example_idx": 56_353,
                "batch_idx": 342_016,
            }
        },
    }

    assert _dataset_state_cursor(reset["dataset_state"]) == (
        1_887_214,
        0,
        0,
        0,
        0,
    )
    assert _usable_iterator_checkpoint(reset, 2_660_334) is None
    assert _usable_iterator_checkpoint(advanced, 2_660_334) is advanced


def test_iterator_cursor_merges_skip_wrapper_and_nested_progress() -> None:
    first = {
        "examples_iterable": {
            "batch_idx": 342_016,
            "examples_iterable": {
                "examples_iterable": {
                    "shard_idx": 13,
                    "shard_example_idx": 56_353,
                },
                "skipped": 1_887_214,
            },
            "num_chunks_since_previous_state": 1_339,
        }
    }
    later = {
        "examples_iterable": {
            "batch_idx": 352_016,
            "examples_iterable": {
                "examples_iterable": {
                    "shard_idx": 13,
                    "shard_example_idx": 66_353,
                },
                "skipped": 1_887_214,
            },
            "num_chunks_since_previous_state": 1_417,
        }
    }

    assert _dataset_state_cursor(first) == (
        1_887_214,
        13,
        56_353,
        342_016,
        1_339,
    )
    assert _dataset_state_cursor(later) > _dataset_state_cursor(first)


def test_closed_huggingface_client_is_a_transient_source_error() -> None:
    error = RuntimeError("Cannot send a request, as the client has been closed.")

    assert _is_transient_source_error(error) is True
    assert _is_transient_source_error(ValueError("invalid source schema")) is False


def test_reset_huggingface_client_evicts_cached_filesystems(monkeypatch) -> None:
    from huggingface_hub import HfFileSystem
    from huggingface_hub import utils as hub_utils

    calls = []
    monkeypatch.setattr(hub_utils, "close_session", lambda: calls.append("client"))
    monkeypatch.setattr(
        HfFileSystem,
        "clear_instance_cache",
        lambda: calls.append("filesystem"),
    )

    public_corpus._reset_huggingface_http_client()

    assert calls == ["client", "filesystem"]


def test_build_restarts_only_after_transient_source_failure(monkeypatch) -> None:
    attempts = []
    delays = []
    resets = []

    def fake_build(*_args, **_kwargs):
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return {"completed": True}

    monkeypatch.setattr(public_corpus, "build", fake_build)
    monkeypatch.setattr(public_corpus.time, "sleep", delays.append)
    monkeypatch.setattr(
        public_corpus, "_reset_huggingface_http_client", lambda: resets.append(True)
    )

    result = build_with_retries(maximum_source_restarts=2)

    assert result == {"completed": True}
    assert len(attempts) == 3
    assert delays == [5, 10]
    assert resets == [True, True]


def test_fingerprint_database_reuses_matching_committed_state(
    tmp_path: Path, monkeypatch
) -> None:
    source_id = "public-source"
    record_id = "record-1"
    document = CanonicalDocument(
        schema_version=1,
        document_id=make_document_id(source_id, record_id),
        source_id=source_id,
        source_record_id=record_id,
        text="A real public document used for fingerprint recovery.",
        language="en",
        content_type="general",
        privacy_warnings=(),
        quality_warnings=(),
        metadata={},
    )
    documents = tmp_path / "documents.jsonl"
    documents.write_text(document.to_json_line() + "\n", encoding="utf-8")
    database = tmp_path / "fingerprints.sqlite3"
    connection = _rebuild_fingerprints(
        documents,
        database,
        committed_output_bytes=documents.stat().st_size,
        records=1,
    )
    connection.close()
    monkeypatch.setattr(
        public_corpus.CanonicalDocument,
        "from_json_line",
        lambda _line: (_ for _ in ()).throw(AssertionError("unexpected rescan")),
    )

    reused = _rebuild_fingerprints(
        documents,
        database,
        committed_output_bytes=documents.stat().st_size,
        records=1,
    )

    assert reused.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0] == 1
    reused.close()
