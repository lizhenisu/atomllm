from pathlib import Path

import pytest

from atomllm.model.config import (
    ModelConfigError,
    calculate_parameter_count,
    load_model_config,
)


CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")
ATOM_50M_CONFIG_PATH = Path("configs/model/atom-50m.yaml")


def write_modified_config(tmp_path: Path, old: str, new: str) -> Path:
    content = CONFIG_PATH.read_text(encoding="utf-8").replace(old, new)
    path = tmp_path / "model.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_committed_model_config_with_exact_parameter_count() -> None:
    config = load_model_config(CONFIG_PATH)
    breakdown = calculate_parameter_count(config)

    assert config.tokenizer.vocab_size == 32_000
    assert config.dimensions.max_sequence_length == 8_192
    assert config.dimensions.num_layers == 24
    assert config.dimensions.hidden_size == 1_024
    assert config.dimensions.num_attention_heads == 16
    assert config.dimensions.num_key_value_heads == 4
    assert config.dimensions.head_dim == 64
    assert config.dimensions.ffn_hidden_size == 2_816
    assert breakdown.token_embeddings == 32_768_000
    assert breakdown.attention_per_layer == 2_621_440
    assert breakdown.attention_all_layers == 62_914_560
    assert breakdown.ffn_per_layer == 8_650_752
    assert breakdown.ffn_all_layers == 207_618_048
    assert breakdown.rmsnorm_per_layer == 2_048
    assert breakdown.rmsnorm_all_layers == 49_152
    assert breakdown.final_rmsnorm == 1_024
    assert breakdown.lm_head == 0
    assert breakdown.total == 303_350_784
    assert breakdown.total == config.expected_parameter_count


def test_atom_50m_has_formal_tokenizer_and_exact_parameter_count() -> None:
    config = load_model_config(ATOM_50M_CONFIG_PATH)
    breakdown = calculate_parameter_count(config)

    assert config.status == "release"
    assert config.tokenizer.version_id == (
        "tokenizer-version-atom-tokenizer-formal-v1-936e948b3509"
    )
    assert config.dimensions.num_layers == 12
    assert config.dimensions.hidden_size == 512
    assert breakdown.total == 50_213_376


def test_rejects_hidden_size_head_mismatch(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "head_dim: 64", "head_dim: 32")

    with pytest.raises(ModelConfigError, match="hidden_size must equal"):
        load_model_config(path)


def test_rejects_invalid_gqa_grouping(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "num_key_value_heads: 4",
        "num_key_value_heads: 3",
    )

    with pytest.raises(ModelConfigError, match="must be divisible"):
        load_model_config(path)


def test_rejects_parameter_count_drift(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "expected_parameter_count: 303350784",
        "expected_parameter_count: 303350785",
    )

    with pytest.raises(ModelConfigError, match="does not match"):
        load_model_config(path)


def test_rejects_changed_core_special_token_id(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "    eos: 3", "    eos: 4")

    with pytest.raises(ModelConfigError, match="Tokenizer v1 protocol"):
        load_model_config(path)


def test_rejects_untied_embeddings(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "tie_word_embeddings: true",
        "tie_word_embeddings: false",
    )

    with pytest.raises(ModelConfigError, match="must be true"):
        load_model_config(path)


def test_rejects_biases(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "use_bias: false", "use_bias: true")

    with pytest.raises(ModelConfigError, match="must be false"):
        load_model_config(path)


def test_rejects_rope_scaling_in_model_v1(tmp_path: Path) -> None:
    path = write_modified_config(tmp_path, "rope_scaling: null", "rope_scaling: linear")

    with pytest.raises(ModelConfigError, match="must be null"):
        load_model_config(path)


def test_rejects_nonzero_dropout(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "attention_dropout: 0.0",
        "attention_dropout: 0.1",
    )

    with pytest.raises(ModelConfigError, match="must be 0.0"):
        load_model_config(path)


def test_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    path = write_modified_config(
        tmp_path,
        "expected_parameter_count: 303350784",
        "expected_parameter_count: 303350784\nextra: value",
    )

    with pytest.raises(ModelConfigError, match="unknown fields: extra"):
        load_model_config(path)
