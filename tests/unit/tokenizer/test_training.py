import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer

import atomllm.tokenizer.training as training
from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.tokenizer.config import load_tokenizer_config
from atomllm.tokenizer.training import (
    TokenizerTrainingError,
    train_tokenizer,
    verify_training_input,
)


CONFIG_PATH = Path("configs/tokenizer/smoke-32k.yaml")


def write_training_data(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True)
    corpus = " ".join(f"synthetic{i:04d}" for i in range(2_000))
    documents = []
    for index in range(2):
        source_id = "synthetic-tokenizer-v1"
        record_id = f"record-{index}"
        documents.append(
            CanonicalDocument.from_mapping(
                {
                    "schema_version": 1,
                    "document_id": make_document_id(source_id, record_id),
                    "source_id": source_id,
                    "source_record_id": record_id,
                    "text": f"{corpus} document{index}",
                    "language": "en",
                    "content_type": "general",
                    "privacy_warnings": [],
                    "quality_warnings": [],
                    "metadata": {"fixture": True},
                }
            )
        )
    path.write_text(
        "".join(f"{document.to_json_line()}\n" for document in documents),
        encoding="utf-8",
    )
    return len(documents), hashlib.sha256(path.read_bytes()).hexdigest()


def small_training_config(project_root: Path):
    data_path = project_root / "data" / "train.jsonl"
    document_count, digest = write_training_data(data_path)
    config = load_tokenizer_config(CONFIG_PATH)
    return replace(
        config,
        algorithm=replace(
            config.algorithm,
            vocab_size=512,
            min_frequency=1,
        ),
        training_data=replace(
            config.training_data,
            data_version_id="data-synthetic-tokenizer-v1-0123456789ab",
            document_count=document_count,
            expected_sha256=digest,
            input_path=Path("data/train.jsonl"),
        ),
        output_dir=Path("artifacts/tokenizer"),
    )


def test_verify_training_input_detects_tampering(tmp_path: Path) -> None:
    config = small_training_config(tmp_path)
    verified = verify_training_input(config, tmp_path)

    assert verified.document_count == 2
    with verified.path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(TokenizerTrainingError, match="SHA-256"):
        verify_training_input(config, tmp_path)


def test_train_save_reload_and_idempotently_verify_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_training_config(tmp_path)
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text("synthetic tokenizer config\n", encoding="utf-8")
    monkeypatch.setattr(training, "load_tokenizer_config", lambda _: config)

    first = train_tokenizer(config_path, tmp_path)
    second = train_tokenizer(config_path, tmp_path)

    output_dir = tmp_path / config.output_dir
    assert second == first
    assert first["vocab_size"] == 512
    assert first["training_eligible"] is False
    assert first["library_versions"]["tokenizer_training_workers"] == 1
    assert first["library_versions"]["tokenizers_parallelism"] == "false"
    assert all(result["roundtrip"] for result in first["probe_results"].values())
    assert all(
        result["unknown_count"] == 0 for result in first["probe_results"].values()
    )
    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    assert tokenizer.token_to_id("<pad>") == 0
    assert tokenizer.token_to_id("<|/think|>") == 14
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "COMPLETED").is_file()

    with (output_dir / "tokenizer.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(TokenizerTrainingError, match="SHA-256 mismatch"):
        train_tokenizer(config_path, tmp_path)


def test_rejects_invalid_training_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_training_config(tmp_path)
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text("synthetic tokenizer config\n", encoding="utf-8")
    monkeypatch.setattr(training, "load_tokenizer_config", lambda _: config)

    with pytest.raises(TokenizerTrainingError, match="workers must be"):
        train_tokenizer(config_path, tmp_path, workers=0)


def test_configures_parallel_training_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

    training._configure_training_workers(4)

    assert training.os.environ["RAYON_NUM_THREADS"] == "4"
    assert training.os.environ["TOKENIZERS_PARALLELISM"] == "true"
