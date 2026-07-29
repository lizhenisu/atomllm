import pytest

from atomllm.tokenizer.public_gpu_selection import (
    PublicGpuSelectionError,
    choose_vocab_size,
)


def test_gpu_gate_keeps_48k_when_effective_text_throughput_improves() -> None:
    selected, effective = choose_vocab_size(
        heldout_selected_vocab_size=48000,
        bytes_per_token_32k=3.0,
        bytes_per_token_48k=3.4,
        tokens_per_second_32k=1000.0,
        tokens_per_second_48k=900.0,
    )

    assert selected == 48000
    assert effective == {"32000": 3000.0, "48000": 3060.0}


def test_gpu_gate_falls_back_to_32k_when_48k_compute_cost_is_too_high() -> None:
    selected, _ = choose_vocab_size(
        heldout_selected_vocab_size=48000,
        bytes_per_token_32k=3.0,
        bytes_per_token_48k=3.4,
        tokens_per_second_32k=1000.0,
        tokens_per_second_48k=800.0,
    )

    assert selected == 32000


def test_gpu_gate_never_overrides_failed_heldout_quality_gate() -> None:
    selected, _ = choose_vocab_size(
        heldout_selected_vocab_size=32000,
        bytes_per_token_32k=3.0,
        bytes_per_token_48k=4.0,
        tokens_per_second_32k=1000.0,
        tokens_per_second_48k=1000.0,
    )

    assert selected == 32000


def test_gpu_gate_rejects_non_positive_measurements() -> None:
    with pytest.raises(PublicGpuSelectionError, match="positive"):
        choose_vocab_size(
            heldout_selected_vocab_size=48000,
            bytes_per_token_32k=3.0,
            bytes_per_token_48k=3.4,
            tokens_per_second_32k=1000.0,
            tokens_per_second_48k=0.0,
        )
