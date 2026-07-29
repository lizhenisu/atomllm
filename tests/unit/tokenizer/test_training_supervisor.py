import os
from pathlib import Path

import pytest

from atomllm.tokenizer.training_supervisor import (
    TokenizerSupervisorError,
    _physical_memory_gib,
    _process_group_rss_bytes,
    _rss_bytes,
)


def test_reads_current_process_rss() -> None:
    rss = _rss_bytes(os.getpid())

    assert isinstance(rss, int)
    assert rss > 0


def test_rejects_missing_proc_process() -> None:
    assert _rss_bytes(2**31 - 1) is None


def test_reads_aggregate_rss_for_current_process_group() -> None:
    rss = _process_group_rss_bytes(os.getpgrp())

    assert isinstance(rss, int)
    assert rss > 0


def test_treats_status_without_rss_as_exited(monkeypatch) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "State:\tZ\n")

    assert _rss_bytes(1234) is None


def test_rejects_invalid_memory_limit(tmp_path) -> None:
    from atomllm.tokenizer.training_supervisor import supervise

    config = tmp_path / "config.yaml"
    config.write_text("x: y\n", encoding="utf-8")
    with pytest.raises(TokenizerSupervisorError, match="maximum_rss_gib"):
        supervise(
            config=config,
            workers=1,
            maximum_rss_gib=_physical_memory_gib() + 1,
            report=tmp_path / "report.json",
        )
