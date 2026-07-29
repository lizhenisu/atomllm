import json

import pytest

from atomllm.tokenizer.training_report_gate import (
    TokenizerTrainingReportGateError,
    verify,
)


def _report(tmp_path, **overrides):
    value = {
        "return_code": 0,
        "memory_limit_exceeded": False,
        "memory_scope": "process_group_aggregate_rss",
        "workers": 256,
        "peak_rss_gib": 448.0,
        **overrides,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_accepts_successful_report_in_memory_range(tmp_path) -> None:
    report = verify(
        _report(tmp_path),
        minimum_rss_gib=440,
        maximum_rss_gib=480,
        expected_workers=256,
    )

    assert report["peak_rss_gib"] == 448.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"return_code": -15}, "return code"),
        ({"memory_limit_exceeded": True}, "memory limit"),
        ({"memory_scope": "parent"}, "memory scope"),
        ({"workers": 128}, "worker count"),
        ({"peak_rss_gib": 480.1}, "RSS peak"),
    ],
)
def test_rejects_failed_contract(tmp_path, override, message) -> None:
    with pytest.raises(TokenizerTrainingReportGateError, match=message):
        verify(
            _report(tmp_path, **override),
            minimum_rss_gib=440,
            maximum_rss_gib=480,
            expected_workers=256,
        )
