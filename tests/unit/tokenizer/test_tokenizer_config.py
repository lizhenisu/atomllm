from pathlib import Path

import pytest

from atomllm.tokenizer.config import (
    EXPECTED_SPECIAL_TOKENS,
    TOKENIZER_MODEL_MAX_LENGTH,
    TOKENIZER_VOCAB_SIZE,
    TokenizerConfigError,
    load_tokenizer_config,
)


CONFIG_PATH = Path("configs/tokenizer/smoke-32k.yaml")


def write_modified_config(tmp_path: Path, old: str, new: str) -> Path:
    content = CONFIG_PATH.read_text(encoding="utf-8").replace(old, new)
    path = tmp_path / "tokenizer.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_committed_smoke_config() -> None:
    config = load_tokenizer_config(CONFIG_PATH)

    assert config.algorithm.vocab_size == TOKENIZER_VOCAB_SIZE
    assert config.model_max_length == TOKENIZER_MODEL_MAX_LENGTH
    assert config.status == "smoke"
    assert config.training_eligible is False
    assert config.training_data.document_count == 980
    assert (
        tuple(
            (token.token_id, token.token, token.purpose)
            for token in config.special_tokens
        )
        == EXPECTED_SPECIAL_TOKENS
    )


def test_rejects_non_32k_vocabulary(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "vocab_size: 32000", "vocab_size: 16000")

    with pytest.raises(TokenizerConfigError, match="vocab_size must be 32000"):
        load_tokenizer_config(path)


def test_rejects_changed_special_token_id(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "  - id: 14\n    token: <|/think|>",
        "  - id: 15\n    token: <|/think|>",
    )

    with pytest.raises(TokenizerConfigError, match="exactly match"):
        load_tokenizer_config(path)


def test_rejects_reordered_special_tokens(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "  - id: 0\n    token: <pad>\n    purpose: padding\n"
        "  - id: 1\n    token: <unk>\n    purpose: unknown",
        "  - id: 1\n    token: <unk>\n    purpose: unknown\n"
        "  - id: 0\n    token: <pad>\n    purpose: padding",
    )

    with pytest.raises(TokenizerConfigError, match="exactly match"):
        load_tokenizer_config(path)


def test_rejects_dropout_for_deterministic_training(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "dropout: 0.0", "dropout: 0.1")

    with pytest.raises(TokenizerConfigError, match="dropout must be 0.0"):
        load_tokenizer_config(path)


def test_rejects_training_eligible_smoke_tokenizer(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "training_eligible: false",
        "training_eligible: true",
    )

    with pytest.raises(TokenizerConfigError, match="must not be training eligible"):
        load_tokenizer_config(path)


def test_rejects_non_training_split(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "  split: train", "  split: validation")

    with pytest.raises(TokenizerConfigError, match="split must be 'train'"):
        load_tokenizer_config(path)


def test_rejects_unsafe_input_path(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "input_path: data/processed/",
        "input_path: ../data/processed/",
    )

    with pytest.raises(TokenizerConfigError, match="safe relative path"):
        load_tokenizer_config(path)


def test_rejects_unknown_evaluation_suite(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "suites: [zh-Hans, en, zh-Hant, ja, code, math, whitespace]",
        "suites: [zh-Hans, en, unsupported]",
    )

    with pytest.raises(TokenizerConfigError, match="unsupported values"):
        load_tokenizer_config(path)


def test_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "output_dir: artifacts/tokenizers/atom-tokenizer-smoke-v1",
        "output_dir: artifacts/tokenizers/atom-tokenizer-smoke-v1\nextra: value",
    )

    with pytest.raises(TokenizerConfigError, match="unknown fields: extra"):
        load_tokenizer_config(path)
