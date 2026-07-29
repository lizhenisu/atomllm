"""Run tokenizer training with an enforced RSS ceiling and peak-memory report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GIB = 1024**3


class TokenizerSupervisorError(RuntimeError):
    """Raised when safe tokenizer training supervision cannot be guaranteed."""


def _physical_memory_gib() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB


def _rss_bytes(pid: int) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise TokenizerSupervisorError("unexpected /proc VmRSS format")
            return int(fields[1]) * 1024
    # A process that exits between ``poll()`` and this read may briefly remain as
    # a zombie.  Linux keeps /proc/<pid>/status for that state but omits VmRSS.
    # Treat it like a disappeared process; ``wait()`` below remains authoritative
    # for the actual exit status.
    return None


def _process_group_rss_bytes(process_group_id: int) -> int | None:
    """Return aggregate RSS for every live process in one process group."""
    total = 0
    found = False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != process_group_id:
                continue
        except ProcessLookupError, PermissionError:
            continue
        rss = _rss_bytes(pid)
        if rss is not None:
            total += rss
            found = True
    return total if found else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def supervise(
    *,
    config: Path,
    workers: int,
    maximum_rss_gib: float,
    report: Path,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    if not config.is_file():
        raise FileNotFoundError(config)
    if type(workers) is not int or workers <= 0:
        raise TokenizerSupervisorError("workers must be positive")
    physical_memory_gib = _physical_memory_gib()
    if not 0 < maximum_rss_gib < physical_memory_gib:
        raise TokenizerSupervisorError(
            "maximum_rss_gib must be positive and below physical memory"
        )
    if not 0.1 <= poll_seconds <= 10:
        raise TokenizerSupervisorError("poll_seconds must be in [0.1, 10]")
    command = [
        sys.executable,
        "-m",
        "atomllm.tokenizer.training",
        "--config",
        str(config),
        "--workers",
        str(workers),
    ]
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    process = subprocess.Popen(command, start_new_session=True)
    peak_rss = 0
    exceeded = False
    limit = int(maximum_rss_gib * GIB)
    try:
        while process.poll() is None:
            rss = _process_group_rss_bytes(process.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
                if rss > limit:
                    exceeded = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
            time.sleep(poll_seconds)
        return_code = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    result = {
        "schema_version": 1,
        "config": str(config),
        "config_sha256": _sha256(config),
        "workers": workers,
        "maximum_rss_gib": maximum_rss_gib,
        "physical_memory_gib": physical_memory_gib,
        "peak_rss_bytes": peak_rss,
        "peak_rss_gib": peak_rss / GIB,
        "memory_scope": "process_group_aggregate_rss",
        "memory_limit_exceeded": exceeded,
        "return_code": return_code,
        "started_at": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "command": command,
    }
    _write_json(report, result)
    if exceeded:
        raise TokenizerSupervisorError(
            f"tokenizer training exceeded {maximum_rss_gib:.1f}GiB RSS"
        )
    if return_code != 0:
        raise TokenizerSupervisorError(
            f"tokenizer training exited with status {return_code}"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--maximum-rss-gib", type=float, default=480.0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = supervise(
        config=args.config,
        workers=args.workers,
        maximum_rss_gib=args.maximum_rss_gib,
        report=args.report,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
