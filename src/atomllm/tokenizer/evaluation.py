"""Independent correctness, compression, and throughput evaluation for tokenizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tokenizers
import yaml
from tokenizers import Tokenizer

from atomllm.tokenizer.config import EXPECTED_SPECIAL_TOKENS


EVALUATION_SCHEMA_VERSION = 1
REQUIRED_SUITE_ORDER = (
    "zh-Hans",
    "en",
    "zh-Hant",
    "ja",
    "code",
    "math",
    "digits",
    "whitespace",
)


class TokenizerEvaluationError(RuntimeError):
    """Raised when evaluation inputs or tokenizer artifacts are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerEvaluationError(f"cannot read {context}: {path.name}") from error
    if not isinstance(value, dict):
        raise TokenizerEvaluationError(f"{context} must be a JSON object")
    return value


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TokenizerEvaluationError(f"{context} must be a mapping")
    return value


def _exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise TokenizerEvaluationError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise TokenizerEvaluationError(
            f"{context} has unknown fields: {', '.join(unknown)}"
        )


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TokenizerEvaluationError(f"{field_name} must be a non-empty string")
    return value


def _safe_path(value: Any, field_name: str) -> Path:
    path = Path(_non_empty_string(value, field_name))
    if path.is_absolute() or ".." in path.parts:
        raise TokenizerEvaluationError(f"{field_name} must be a safe relative path")
    return path


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmup_iterations: int
    measured_iterations: int
    batch_repetitions: int

    @classmethod
    def from_mapping(cls, value: Any) -> BenchmarkConfig:
        data = _mapping(value, "benchmark")
        _exact_keys(
            data,
            {"warmup_iterations", "measured_iterations", "batch_repetitions"},
            "benchmark",
        )
        for field_name in (
            "warmup_iterations",
            "measured_iterations",
            "batch_repetitions",
        ):
            if type(data[field_name]) is not int or data[field_name] <= 0:
                raise TokenizerEvaluationError(
                    f"benchmark.{field_name} must be a positive integer"
                )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TokenizerEvaluationConfig:
    schema_version: int
    name: str
    tokenizer_artifact: Path
    expected_artifact_id: str
    expected_tokenizer_sha256: str
    benchmark: BenchmarkConfig
    suites: dict[str, tuple[str, ...]]
    output_dir: Path

    @classmethod
    def from_mapping(cls, value: Any) -> TokenizerEvaluationConfig:
        data = _mapping(value, "evaluation config")
        _exact_keys(
            data,
            {
                "schema_version",
                "name",
                "tokenizer_artifact",
                "expected_artifact_id",
                "expected_tokenizer_sha256",
                "benchmark",
                "suites",
                "output_dir",
            },
            "evaluation config",
        )
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != EVALUATION_SCHEMA_VERSION
        ):
            raise TokenizerEvaluationError(
                f"schema_version must be {EVALUATION_SCHEMA_VERSION}"
            )
        name = _non_empty_string(data["name"], "name")
        artifact_id = _non_empty_string(
            data["expected_artifact_id"], "expected_artifact_id"
        )
        tokenizer_sha256 = data["expected_tokenizer_sha256"]
        if (
            not isinstance(tokenizer_sha256, str)
            or len(tokenizer_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in tokenizer_sha256
            )
        ):
            raise TokenizerEvaluationError(
                "expected_tokenizer_sha256 must be 64 lowercase hex digits"
            )
        raw_suites = _mapping(data["suites"], "suites")
        if tuple(raw_suites) != REQUIRED_SUITE_ORDER:
            raise TokenizerEvaluationError(
                "suites must exactly match the required evaluation order"
            )
        suites: dict[str, tuple[str, ...]] = {}
        for suite_name, raw_samples in raw_suites.items():
            if (
                not isinstance(raw_samples, list)
                or not raw_samples
                or not all(isinstance(sample, str) and sample for sample in raw_samples)
            ):
                raise TokenizerEvaluationError(
                    f"suites.{suite_name} must be a non-empty string list"
                )
            if len(raw_samples) != len(set(raw_samples)):
                raise TokenizerEvaluationError(
                    f"suites.{suite_name} must not contain duplicate samples"
                )
            suites[suite_name] = tuple(raw_samples)
        return cls(
            schema_version=EVALUATION_SCHEMA_VERSION,
            name=name,
            tokenizer_artifact=_safe_path(
                data["tokenizer_artifact"], "tokenizer_artifact"
            ),
            expected_artifact_id=artifact_id,
            expected_tokenizer_sha256=tokenizer_sha256,
            benchmark=BenchmarkConfig.from_mapping(data["benchmark"]),
            suites=suites,
            output_dir=_safe_path(data["output_dir"], "output_dir"),
        )


def load_evaluation_config(path: str | Path) -> TokenizerEvaluationConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"tokenizer evaluation config not found: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise TokenizerEvaluationError(f"invalid evaluation YAML: {error}") from error
    return TokenizerEvaluationConfig.from_mapping(value)


def _resolve_within_root(root: Path, relative_path: Path, field_name: str) -> Path:
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise TokenizerEvaluationError(f"{field_name} resolves outside project root")
    return resolved


def verify_tokenizer_artifact(
    config: TokenizerEvaluationConfig,
    project_root: str | Path,
) -> tuple[Tokenizer, dict[str, Any], Path]:
    root = Path(project_root).resolve()
    artifact_dir = _resolve_within_root(
        root,
        config.tokenizer_artifact,
        "tokenizer_artifact",
    )
    tokenizer, manifest, manifest_path = verify_tokenizer_directory(artifact_dir)
    if manifest.get("artifact_id") != config.expected_artifact_id:
        raise TokenizerEvaluationError("tokenizer artifact ID does not match config")
    tokenizer_path = artifact_dir / "tokenizer.json"
    if _sha256(tokenizer_path) != config.expected_tokenizer_sha256:
        raise TokenizerEvaluationError("tokenizer.json SHA-256 does not match config")
    return tokenizer, manifest, manifest_path


def verify_tokenizer_directory(
    artifact_dir: str | Path,
) -> tuple[Tokenizer, dict[str, Any], Path]:
    """Verify a completed tokenizer directory without pinning a caller's identity."""
    directory = Path(artifact_dir)
    manifest_path = directory / "manifest.json"
    completed_path = directory / "COMPLETED"
    if not manifest_path.is_file() or not completed_path.is_file():
        raise TokenizerEvaluationError("tokenizer artifact is incomplete")
    manifest = _read_json(manifest_path, "tokenizer manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TokenizerEvaluationError("tokenizer manifest has invalid files")
    for name, metadata in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(metadata, dict)
        ):
            raise TokenizerEvaluationError("tokenizer file metadata is invalid")
        file_path = directory / name
        if not file_path.is_file() or _sha256(file_path) != metadata.get("sha256"):
            raise TokenizerEvaluationError(
                f"tokenizer artifact file SHA-256 mismatch: {name}"
            )
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise TokenizerEvaluationError("tokenizer.json is missing")
    if completed_path.read_text(encoding="utf-8") != (
        f"{_sha256(manifest_path)}  manifest.json\n"
    ):
        raise TokenizerEvaluationError("tokenizer COMPLETED marker is invalid")
    return Tokenizer.from_file(str(tokenizer_path)), manifest, manifest_path


def _rate(count: int, elapsed_ns: int) -> float:
    return round(count / (elapsed_ns / 1_000_000_000), 2)


def evaluate_suite(
    tokenizer: Tokenizer,
    samples: tuple[str, ...],
    benchmark: BenchmarkConfig,
    *,
    expected_atomic_ids: tuple[int, ...] | None = None,
    expected_unknown_count: int = 0,
) -> dict[str, Any]:
    normalized_samples = tuple(
        unicodedata.normalize("NFC", sample) for sample in samples
    )
    encodings = tokenizer.encode_batch(
        list(normalized_samples),
        add_special_tokens=False,
    )
    token_ids = [encoding.ids for encoding in encodings]
    decoded = tokenizer.decode_batch(token_ids, skip_special_tokens=False)
    roundtrip_failures = sum(
        actual != expected
        for actual, expected in zip(decoded, normalized_samples, strict=True)
    )
    unknown_id = tokenizer.token_to_id("<unk>")
    if unknown_id is None:
        raise TokenizerEvaluationError("tokenizer does not define <unk>")
    unknown_count = sum(ids.count(unknown_id) for ids in token_ids)
    unexpected_unknown_count = unknown_count - expected_unknown_count
    if unexpected_unknown_count < 0:
        raise TokenizerEvaluationError("expected unknown count exceeds observed count")
    token_count = sum(len(ids) for ids in token_ids)
    character_count = sum(len(sample) for sample in normalized_samples)
    utf8_bytes = sum(len(sample.encode("utf-8")) for sample in normalized_samples)
    atomic_failures = 0
    if expected_atomic_ids is not None:
        atomic_failures = sum(
            ids != [expected_id]
            for ids, expected_id in zip(token_ids, expected_atomic_ids, strict=True)
        )

    benchmark_samples = list(normalized_samples) * benchmark.batch_repetitions
    benchmark_encodings = tokenizer.encode_batch(
        benchmark_samples,
        add_special_tokens=False,
    )
    benchmark_ids = [encoding.ids for encoding in benchmark_encodings]
    for _ in range(benchmark.warmup_iterations):
        tokenizer.encode_batch(benchmark_samples, add_special_tokens=False)
        tokenizer.decode_batch(benchmark_ids, skip_special_tokens=False)

    start = time.perf_counter_ns()
    for _ in range(benchmark.measured_iterations):
        tokenizer.encode_batch(benchmark_samples, add_special_tokens=False)
    encode_elapsed_ns = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    for _ in range(benchmark.measured_iterations):
        tokenizer.decode_batch(benchmark_ids, skip_special_tokens=False)
    decode_elapsed_ns = time.perf_counter_ns() - start

    repetitions = benchmark.batch_repetitions * benchmark.measured_iterations
    return {
        "sample_count": len(samples),
        "character_count": character_count,
        "utf8_bytes": utf8_bytes,
        "token_count": token_count,
        "tokens_per_character": round(token_count / character_count, 6),
        "characters_per_token": round(character_count / token_count, 6),
        "bytes_per_token": round(utf8_bytes / token_count, 6),
        "unknown_count": unknown_count,
        "expected_unknown_count": expected_unknown_count,
        "unexpected_unknown_count": unexpected_unknown_count,
        "unknown_rate": round(unexpected_unknown_count / token_count, 8),
        "roundtrip_failures": roundtrip_failures,
        "atomic_failures": atomic_failures,
        "throughput": {
            "encode_texts_per_second": _rate(
                len(samples) * repetitions, encode_elapsed_ns
            ),
            "encode_characters_per_second": _rate(
                character_count * repetitions,
                encode_elapsed_ns,
            ),
            "encode_tokens_per_second": _rate(
                token_count * repetitions,
                encode_elapsed_ns,
            ),
            "decode_texts_per_second": _rate(
                len(samples) * repetitions, decode_elapsed_ns
            ),
            "decode_tokens_per_second": _rate(
                token_count * repetitions,
                decode_elapsed_ns,
            ),
        },
    }


def _validate_existing_report(
    output_dir: Path,
    config_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    report_path = output_dir / "report.json"
    completed_path = output_dir / "COMPLETED"
    config_path = output_dir / "config.yaml"
    if not all(path.is_file() for path in (report_path, completed_path, config_path)):
        raise TokenizerEvaluationError("existing evaluation artifact is incomplete")
    report = _read_json(report_path, "evaluation report")
    if report.get("config_sha256") != config_sha256:
        raise TokenizerEvaluationError("existing evaluation uses a different config")
    if report.get("tokenizer_sha256") != tokenizer_sha256:
        raise TokenizerEvaluationError("existing evaluation uses a different tokenizer")
    if _sha256(config_path) != config_sha256:
        raise TokenizerEvaluationError("evaluation config snapshot SHA-256 mismatch")
    if completed_path.read_text(encoding="utf-8") != (
        f"{_sha256(report_path)}  report.json\n"
    ):
        raise TokenizerEvaluationError("evaluation COMPLETED marker is invalid")
    return report


def evaluate_tokenizer_artifact(
    config_path: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = (root / resolved_config_path).resolve()
    config = load_evaluation_config(resolved_config_path)
    config_sha256 = _sha256(resolved_config_path)
    tokenizer, tokenizer_manifest, tokenizer_manifest_path = verify_tokenizer_artifact(
        config,
        root,
    )
    output_dir = _resolve_within_root(root, config.output_dir, "output_dir")
    existing = _validate_existing_report(
        output_dir,
        config_sha256,
        config.expected_tokenizer_sha256,
    )
    if existing is not None:
        return existing

    suite_results = {
        suite_name: evaluate_suite(tokenizer, samples, config.benchmark)
        for suite_name, samples in config.suites.items()
    }
    special_samples = tuple(token for _, token, _ in EXPECTED_SPECIAL_TOKENS)
    special_ids = tuple(token_id for token_id, _, _ in EXPECTED_SPECIAL_TOKENS)
    suite_results["special_tokens"] = evaluate_suite(
        tokenizer,
        special_samples,
        config.benchmark,
        expected_atomic_ids=special_ids,
        expected_unknown_count=1,
    )
    total_unknowns = sum(
        result["unexpected_unknown_count"] for result in suite_results.values()
    )
    total_roundtrip_failures = sum(
        result["roundtrip_failures"] for result in suite_results.values()
    )
    total_atomic_failures = suite_results["special_tokens"]["atomic_failures"]
    if total_unknowns or total_roundtrip_failures or total_atomic_failures:
        raise TokenizerEvaluationError("tokenizer correctness evaluation failed")

    report_payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "name": config.name,
        "config_sha256": config_sha256,
        "tokenizer_artifact_id": tokenizer_manifest["artifact_id"],
        "tokenizer_manifest_sha256": _sha256(tokenizer_manifest_path),
        "tokenizer_sha256": config.expected_tokenizer_sha256,
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "benchmark": {
            "warmup_iterations": config.benchmark.warmup_iterations,
            "measured_iterations": config.benchmark.measured_iterations,
            "batch_repetitions": config.benchmark.batch_repetitions,
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
        },
        "summary": {
            "suite_count": len(suite_results),
            "sample_count": sum(
                result["sample_count"] for result in suite_results.values()
            ),
            "unknown_count": total_unknowns,
            "roundtrip_failures": total_roundtrip_failures,
            "special_token_atomic_failures": total_atomic_failures,
            "all_correctness_checks_passed": True,
        },
        "suites": suite_results,
    }
    identity = json.dumps(
        report_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    report_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    report = {
        **report_payload,
        "evaluation_id": f"tokenizer-eval-{config.name}-{report_digest[:12]}",
        "identity_sha256": report_digest,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        report_path = temporary_dir / "report.json"
        report_path.write_text(
            f"{json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)}\n",
            encoding="utf-8",
        )
        shutil.copyfile(resolved_config_path, temporary_dir / "config.yaml")
        (temporary_dir / "COMPLETED").write_text(
            f"{_sha256(report_path)}  report.json\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AtomLLM tokenizer correctness, compression, and throughput."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/evaluation-v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_tokenizer_artifact(args.config, args.project_root)
    summary = report["summary"]
    print(
        "Tokenizer evaluation complete: "
        f"{report['evaluation_id']}, "
        f"suites={summary['suite_count']}, "
        f"unknowns={summary['unknown_count']}, "
        f"roundtrip_failures={summary['roundtrip_failures']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
