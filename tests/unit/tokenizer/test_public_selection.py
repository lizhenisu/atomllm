import json

import pytest

from atomllm.tokenizer.public_selection import (
    PublicTokenizerSelectionError,
    _reduction,
    _verified_memory_report,
)


def test_token_reduction() -> None:
    assert _reduction(100, 88) == pytest.approx(0.12)


@pytest.mark.parametrize(("baseline", "candidate"), [(0, 1), (1, 0), (-1, 1)])
def test_token_reduction_requires_positive_counts(
    baseline: int, candidate: int
) -> None:
    with pytest.raises(PublicTokenizerSelectionError, match="positive"):
        _reduction(baseline, candidate)


def test_memory_gate_requires_near_full_host_utilization(tmp_path) -> None:
    report = tmp_path / "memory.json"
    report.write_text(
        json.dumps(
            {
                "return_code": 0,
                "memory_limit_exceeded": False,
                "maximum_rss_gib": 480.0,
                "peak_rss_gib": 430.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PublicTokenizerSelectionError, match="peak RSS"):
        _verified_memory_report(report, 480.0, 440.0)
