import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from atomllm.tokenizer.config import load_tokenizer_config
from atomllm.tokenizer.evaluation import (
    BenchmarkConfig,
    TokenizerEvaluationError,
    evaluate_suite,
    load_evaluation_config,
    verify_tokenizer_artifact,
)
from atomllm.tokenizer.training import _build_tokenizer


EVALUATION_CONFIG_PATH = Path("configs/tokenizer/evaluation-v1.yaml")
TOKENIZER_CONFIG_PATH = Path("configs/tokenizer/smoke-32k.yaml")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_small_tokenizer():
    config = load_tokenizer_config(TOKENIZER_CONFIG_PATH)
    config = replace(
        config,
        algorithm=replace(
            config.algorithm,
            vocab_size=512,
            min_frequency=1,
        ),
    )
    tokenizer, trainer = _build_tokenizer(config)
    tokenizer.train_from_iterator(
        [
            "合成中文文本 synthetic English text 12345",
            "繁體中文 日本語 code = value + 1",
        ],
        trainer=trainer,
        length=2,
    )
    return tokenizer


def test_loads_committed_evaluation_config() -> None:
    config = load_evaluation_config(EVALUATION_CONFIG_PATH)

    assert config.benchmark.measured_iterations == 100
    assert tuple(config.suites) == (
        "zh-Hans",
        "en",
        "zh-Hant",
        "ja",
        "code",
        "math",
        "digits",
        "whitespace",
    )
    assert all(len(samples) == 4 for samples in config.suites.values())


def test_rejects_missing_required_suite(tmp_path: Path) -> None:
    content = EVALUATION_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  digits:\n",
        "  unsupported:\n",
    )
    path = tmp_path / "evaluation.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(TokenizerEvaluationError, match="required evaluation order"):
        load_evaluation_config(path)


def test_rejects_non_positive_benchmark_iterations(tmp_path: Path) -> None:
    content = EVALUATION_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "measured_iterations: 100",
        "measured_iterations: 0",
    )
    path = tmp_path / "evaluation.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(TokenizerEvaluationError, match="must be a positive integer"):
        load_evaluation_config(path)


def test_evaluate_suite_reports_roundtrip_compression_and_throughput() -> None:
    tokenizer = make_small_tokenizer()
    result = evaluate_suite(
        tokenizer,
        ("合成文本。", "Hello 123!"),
        BenchmarkConfig(
            warmup_iterations=1,
            measured_iterations=1,
            batch_repetitions=1,
        ),
    )

    assert result["sample_count"] == 2
    assert result["token_count"] > 0
    assert result["bytes_per_token"] > 0
    assert result["unknown_count"] == 0
    assert result["unexpected_unknown_count"] == 0
    assert result["roundtrip_failures"] == 0
    assert result["throughput"]["encode_tokens_per_second"] > 0
    assert result["throughput"]["decode_tokens_per_second"] > 0


def test_special_tokens_remain_atomic() -> None:
    tokenizer = make_small_tokenizer()
    result = evaluate_suite(
        tokenizer,
        ("<pad>", "<unk>", "<|assistant|>", "<|/think|>"),
        BenchmarkConfig(
            warmup_iterations=1,
            measured_iterations=1,
            batch_repetitions=1,
        ),
        expected_atomic_ids=(0, 1, 6, 14),
        expected_unknown_count=1,
    )

    assert result["atomic_failures"] == 0
    assert result["unknown_count"] == 1
    assert result["unexpected_unknown_count"] == 0
    assert result["roundtrip_failures"] == 0


def test_verify_tokenizer_artifact_detects_file_tampering(tmp_path: Path) -> None:
    tokenizer = make_small_tokenizer()
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    manifest = {
        "artifact_id": "tokenizer-synthetic-eval-v1-0123456789ab",
        "files": {
            "tokenizer.json": {
                "bytes": tokenizer_path.stat().st_size,
                "sha256": sha256(tokenizer_path),
            }
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        f"{json.dumps(manifest, sort_keys=True)}\n",
        encoding="utf-8",
    )
    (artifact_dir / "COMPLETED").write_text(
        f"{sha256(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )
    config = load_evaluation_config(EVALUATION_CONFIG_PATH)
    config = replace(
        config,
        tokenizer_artifact=Path("artifact"),
        expected_artifact_id=manifest["artifact_id"],
        expected_tokenizer_sha256=sha256(tokenizer_path),
    )

    loaded, loaded_manifest, _ = verify_tokenizer_artifact(config, tmp_path)

    assert loaded.get_vocab_size() == tokenizer.get_vocab_size()
    assert loaded_manifest == manifest

    with tokenizer_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(TokenizerEvaluationError, match="SHA-256 mismatch"):
        verify_tokenizer_artifact(config, tmp_path)
