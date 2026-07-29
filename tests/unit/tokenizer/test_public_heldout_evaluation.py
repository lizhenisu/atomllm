from atomllm.tokenizer.public_heldout_evaluation import (
    _reported,
)


def test_reported_metrics_calculates_compression() -> None:
    result = _reported(
        {
            "document_count": 2,
            "character_count": 100,
            "utf8_bytes": 160,
            "token_count": 40,
            "unknown_count": 0,
            "roundtrip_failures": 0,
        }
    )

    assert result["characters_per_token"] == 2.5
    assert result["bytes_per_token"] == 4.0
