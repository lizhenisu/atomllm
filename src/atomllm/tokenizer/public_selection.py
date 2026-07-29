"""Select a public tokenizer candidate from matched held-out evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class PublicTokenizerSelectionError(RuntimeError):
    """Raised when tokenizer candidates cannot be compared safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicTokenizerSelectionError(
            f"cannot read JSON artifact: {path}"
        ) from error
    if not isinstance(value, dict):
        raise PublicTokenizerSelectionError(f"JSON artifact must be an object: {path}")
    return value


def _verified_evaluation(directory: Path) -> tuple[dict[str, Any], str]:
    report_path = directory / "report.json"
    completed_path = directory / "COMPLETED"
    if not report_path.is_file() or not completed_path.is_file():
        raise PublicTokenizerSelectionError(f"evaluation is incomplete: {directory}")
    report_sha = _sha256(report_path)
    if completed_path.read_text(encoding="utf-8") != f"{report_sha}  report.json\n":
        raise PublicTokenizerSelectionError(
            f"evaluation marker is invalid: {directory}"
        )
    report = _read_json(report_path)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise PublicTokenizerSelectionError("evaluation summary is invalid")
    if summary.get("unknown_count") != 0 or summary.get("roundtrip_failures") != 0:
        raise PublicTokenizerSelectionError("candidate failed tokenizer correctness")
    return report, report_sha


def _verified_memory_report(
    path: Path,
    expected_limit_gib: float,
    minimum_peak_gib: float,
) -> dict[str, Any]:
    report = _read_json(path)
    if (
        report.get("return_code") != 0
        or report.get("memory_limit_exceeded") is not False
    ):
        raise PublicTokenizerSelectionError(f"candidate training failed: {path}")
    maximum = report.get("maximum_rss_gib")
    peak = report.get("peak_rss_gib")
    if type(maximum) not in {int, float} or float(maximum) != expected_limit_gib:
        raise PublicTokenizerSelectionError("candidate used an unexpected RSS limit")
    if (
        type(peak) not in {int, float}
        or not minimum_peak_gib <= float(peak) <= expected_limit_gib
    ):
        raise PublicTokenizerSelectionError("candidate peak RSS is invalid")
    return report


def _reduction(baseline_tokens: int, candidate_tokens: int) -> float:
    if baseline_tokens <= 0 or candidate_tokens <= 0:
        raise PublicTokenizerSelectionError("candidate token count must be positive")
    return 1.0 - candidate_tokens / baseline_tokens


def select(
    *,
    evaluation_32k_dir: Path,
    evaluation_48k_dir: Path,
    memory_32k_report: Path,
    memory_48k_report: Path,
    output_dir: Path,
    hidden_size: int = 1024,
    minimum_total_token_reduction: float = 0.10,
    minimum_rss_gib: float = 440.0,
    maximum_rss_gib: float = 480.0,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    if type(hidden_size) is not int or hidden_size <= 0:
        raise PublicTokenizerSelectionError("hidden_size must be positive")
    if not 0 < minimum_total_token_reduction < 1:
        raise PublicTokenizerSelectionError(
            "minimum_total_token_reduction must be in (0, 1)"
        )
    if not 0 < minimum_rss_gib < maximum_rss_gib:
        raise PublicTokenizerSelectionError(
            "minimum_rss_gib must be positive and below maximum_rss_gib"
        )
    root = project_root.resolve()

    def resolved(path: Path) -> Path:
        value = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not value.is_relative_to(root):
            raise PublicTokenizerSelectionError("selection path escapes project root")
        return value

    eval_32, eval_32_sha = _verified_evaluation(resolved(evaluation_32k_dir))
    eval_48, eval_48_sha = _verified_evaluation(resolved(evaluation_48k_dir))
    memory_32_path = resolved(memory_32k_report)
    memory_48_path = resolved(memory_48k_report)
    memory_32 = _verified_memory_report(
        memory_32_path, maximum_rss_gib, minimum_rss_gib
    )
    memory_48 = _verified_memory_report(
        memory_48_path, maximum_rss_gib, minimum_rss_gib
    )
    for key in ("snapshot_manifest_sha256", "heldout_sha256"):
        if eval_32.get(key) != eval_48.get(key):
            raise PublicTokenizerSelectionError(
                f"candidate evaluations do not share {key}"
            )
    if eval_32.get("vocab_size") != 32000 or eval_48.get("vocab_size") != 48000:
        raise PublicTokenizerSelectionError("candidate vocab sizes are not 32K and 48K")
    summary_32 = eval_32["summary"]
    summary_48 = eval_48["summary"]
    total_reduction = _reduction(summary_32["token_count"], summary_48["token_count"])
    by_language_reduction = {}
    for language in ("en", "code", "zh-Hans"):
        group_32 = eval_32.get("by_language", {}).get(language)
        group_48 = eval_48.get("by_language", {}).get(language)
        if not isinstance(group_32, dict) or not isinstance(group_48, dict):
            raise PublicTokenizerSelectionError(
                f"candidate is missing held-out language: {language}"
            )
        by_language_reduction[language] = _reduction(
            group_32["token_count"], group_48["token_count"]
        )
    larger_candidate_regresses = any(
        reduction < 0 for reduction in by_language_reduction.values()
    )
    selected_vocab_size = (
        48000
        if total_reduction >= minimum_total_token_reduction
        and not larger_candidate_regresses
        else 32000
    )
    added_parameters = (48000 - 32000) * hidden_size
    report = {
        "schema_version": 1,
        "selection_version": "public-tokenizer-selection-v1",
        "selected_vocab_size": selected_vocab_size,
        "requires_gpu_throughput_confirmation": True,
        "minimum_total_token_reduction": minimum_total_token_reduction,
        "required_peak_rss_gib_range": [minimum_rss_gib, maximum_rss_gib],
        "observed_total_token_reduction": round(total_reduction, 8),
        "observed_token_reduction_by_language": {
            key: round(value, 8) for key, value in by_language_reduction.items()
        },
        "larger_candidate_added_tied_embedding_parameters": added_parameters,
        "hidden_size": hidden_size,
        "candidate_peak_rss_gib": {
            "32000": memory_32["peak_rss_gib"],
            "48000": memory_48["peak_rss_gib"],
        },
        "candidate_encode_tokens_per_second": {
            "32000": summary_32["encode_tokens_per_second"],
            "48000": summary_48["encode_tokens_per_second"],
        },
        "lineage": {
            "evaluation_32k_sha256": eval_32_sha,
            "evaluation_48k_sha256": eval_48_sha,
            "memory_32k_sha256": _sha256(memory_32_path),
            "memory_48k_sha256": _sha256(memory_48_path),
            "heldout_sha256": eval_32["heldout_sha256"],
        },
        "checks": {
            "same_heldout_data": True,
            "zero_unknown_tokens": True,
            "zero_roundtrip_failures": True,
            "training_within_memory_limit": True,
            "model_external_capability": False,
        },
    }
    output = resolved(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    (output / "COMPLETED").write_text(
        f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-32k-dir", type=Path, required=True)
    parser.add_argument("--evaluation-48k-dir", type=Path, required=True)
    parser.add_argument("--memory-32k-report", type=Path, required=True)
    parser.add_argument("--memory-48k-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--minimum-total-token-reduction", type=float, default=0.10)
    parser.add_argument("--minimum-rss-gib", type=float, default=440.0)
    parser.add_argument("--maximum-rss-gib", type=float, default=480.0)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = select(
        evaluation_32k_dir=args.evaluation_32k_dir,
        evaluation_48k_dir=args.evaluation_48k_dir,
        memory_32k_report=args.memory_32k_report,
        memory_48k_report=args.memory_48k_report,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        minimum_total_token_reduction=args.minimum_total_token_reduction,
        minimum_rss_gib=args.minimum_rss_gib,
        maximum_rss_gib=args.maximum_rss_gib,
        project_root=args.project_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
