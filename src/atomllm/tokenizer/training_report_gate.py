"""Validate a completed tokenizer training report before releasing dependants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class TokenizerTrainingReportGateError(RuntimeError):
    """Raised when a tokenizer training report does not pass its hard gate."""


def verify(
    report_path: Path,
    *,
    minimum_rss_gib: float,
    maximum_rss_gib: float,
    expected_workers: int,
) -> dict[str, Any]:
    if not 0 < minimum_rss_gib <= maximum_rss_gib:
        raise TokenizerTrainingReportGateError("RSS gate range is invalid")
    if expected_workers <= 0:
        raise TokenizerTrainingReportGateError("expected_workers must be positive")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerTrainingReportGateError(
            "training report is unreadable"
        ) from error
    if not isinstance(report, dict):
        raise TokenizerTrainingReportGateError("training report must be an object")
    peak = report.get("peak_rss_gib")
    checks = {
        "return code": report.get("return_code") == 0,
        "memory limit": report.get("memory_limit_exceeded") is False,
        "memory scope": report.get("memory_scope") == "process_group_aggregate_rss",
        "worker count": report.get("workers") == expected_workers,
        "RSS peak": type(peak) in {int, float}
        and minimum_rss_gib <= float(peak) <= maximum_rss_gib,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise TokenizerTrainingReportGateError(
            "tokenizer training report failed: " + ", ".join(failed)
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-rss-gib", type=float, required=True)
    parser.add_argument("--maximum-rss-gib", type=float, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    args = parser.parse_args(argv)
    report = verify(
        args.report,
        minimum_rss_gib=args.minimum_rss_gib,
        maximum_rss_gib=args.maximum_rss_gib,
        expected_workers=args.expected_workers,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
