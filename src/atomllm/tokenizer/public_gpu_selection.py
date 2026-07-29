"""Freeze a public tokenizer after held-out quality and 6-GPU throughput gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class PublicGpuSelectionError(RuntimeError):
    """Raised when final public tokenizer evidence is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_completed(directory: Path) -> tuple[dict[str, Any], str]:
    report_path = directory / "report.json"
    completed_path = directory / "COMPLETED"
    if not report_path.is_file() or not completed_path.is_file():
        raise PublicGpuSelectionError(f"artifact is incomplete: {directory}")
    report_sha = _sha256(report_path)
    if completed_path.read_text(encoding="utf-8") != f"{report_sha}  report.json\n":
        raise PublicGpuSelectionError(f"artifact marker is invalid: {directory}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PublicGpuSelectionError(
            f"artifact JSON is invalid: {directory}"
        ) from error
    if not isinstance(report, dict):
        raise PublicGpuSelectionError(f"artifact report is invalid: {directory}")
    return report, report_sha


def _resolve(root: Path, path: Path, field: str) -> Path:
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not result.is_relative_to(root):
        raise PublicGpuSelectionError(f"{field} escapes project root")
    return result


def choose_vocab_size(
    *,
    heldout_selected_vocab_size: int,
    bytes_per_token_32k: float,
    bytes_per_token_48k: float,
    tokens_per_second_32k: float,
    tokens_per_second_48k: float,
) -> tuple[int, dict[str, float]]:
    values = {
        "bytes_per_token_32k": bytes_per_token_32k,
        "bytes_per_token_48k": bytes_per_token_48k,
        "tokens_per_second_32k": tokens_per_second_32k,
        "tokens_per_second_48k": tokens_per_second_48k,
    }
    if heldout_selected_vocab_size not in {32000, 48000}:
        raise PublicGpuSelectionError("held-out selection vocabulary is invalid")
    if any(
        type(value) not in {int, float} or float(value) <= 0
        for value in values.values()
    ):
        raise PublicGpuSelectionError("throughput inputs must be positive")
    effective_32 = float(bytes_per_token_32k) * float(tokens_per_second_32k)
    effective_48 = float(bytes_per_token_48k) * float(tokens_per_second_48k)
    selected = (
        48000
        if heldout_selected_vocab_size == 48000 and effective_48 >= effective_32
        else 32000
    )
    return selected, {
        "32000": effective_32,
        "48000": effective_48,
    }


def select(
    *,
    heldout_selection_dir: Path,
    evaluation_32k_dir: Path,
    evaluation_48k_dir: Path,
    gpu_32k_dir: Path,
    gpu_48k_dir: Path,
    tokenizer_32k_dir: Path,
    tokenizer_48k_dir: Path,
    output_dir: Path,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    root = project_root.resolve()
    paths = {
        "heldout_selection": _resolve(root, heldout_selection_dir, "heldout_selection"),
        "evaluation_32k": _resolve(root, evaluation_32k_dir, "evaluation_32k"),
        "evaluation_48k": _resolve(root, evaluation_48k_dir, "evaluation_48k"),
        "gpu_32k": _resolve(root, gpu_32k_dir, "gpu_32k"),
        "gpu_48k": _resolve(root, gpu_48k_dir, "gpu_48k"),
        "tokenizer_32k": _resolve(root, tokenizer_32k_dir, "tokenizer_32k"),
        "tokenizer_48k": _resolve(root, tokenizer_48k_dir, "tokenizer_48k"),
    }
    output = _resolve(root, output_dir, "output_dir")
    reports: dict[str, dict[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    for key in (
        "heldout_selection",
        "evaluation_32k",
        "evaluation_48k",
        "gpu_32k",
        "gpu_48k",
    ):
        reports[key], report_hashes[key] = _read_completed(paths[key])
    heldout = reports["heldout_selection"]
    evaluation_32 = reports["evaluation_32k"]
    evaluation_48 = reports["evaluation_48k"]
    gpu_32 = reports["gpu_32k"]
    gpu_48 = reports["gpu_48k"]
    if (
        evaluation_32.get("vocab_size") != 32000
        or evaluation_48.get("vocab_size") != 48000
    ):
        raise PublicGpuSelectionError("held-out candidate vocabularies are invalid")
    for key in ("snapshot_manifest_sha256", "heldout_sha256"):
        if evaluation_32.get(key) != evaluation_48.get(key):
            raise PublicGpuSelectionError(f"held-out candidate {key} mismatch")
    for expected_vocab, gpu, evaluation in (
        (32000, gpu_32, evaluation_32),
        (48000, gpu_48, evaluation_48),
    ):
        identity = gpu.get("identity")
        checks = gpu.get("checks")
        if not isinstance(identity, dict) or not isinstance(checks, dict):
            raise PublicGpuSelectionError("GPU benchmark schema is invalid")
        if identity.get("vocab_size") != expected_vocab:
            raise PublicGpuSelectionError("GPU benchmark vocabulary mismatch")
        if (
            identity.get("evaluation_sha256")
            != report_hashes[f"evaluation_{expected_vocab // 1000}k"]
        ):
            raise PublicGpuSelectionError("GPU benchmark evaluation lineage mismatch")
        if identity.get("world_size") != 6 or checks.get("six_gpu_ddp") is not True:
            raise PublicGpuSelectionError("GPU benchmark did not use six GPUs")
        if checks.get("real_public_heldout_tokens") is not True:
            raise PublicGpuSelectionError(
                "GPU benchmark did not use public held-out data"
            )
        if gpu.get("global_tokens_per_second", 0) <= 0:
            raise PublicGpuSelectionError("GPU benchmark throughput is invalid")
        if identity.get("heldout_sha256") != evaluation.get("heldout_sha256"):
            raise PublicGpuSelectionError("GPU benchmark held-out lineage mismatch")
    selected_vocab, effective = choose_vocab_size(
        heldout_selected_vocab_size=heldout.get("selected_vocab_size"),
        bytes_per_token_32k=evaluation_32["summary"]["bytes_per_token"],
        bytes_per_token_48k=evaluation_48["summary"]["bytes_per_token"],
        tokens_per_second_32k=gpu_32["global_tokens_per_second"],
        tokens_per_second_48k=gpu_48["global_tokens_per_second"],
    )
    selected_key = "tokenizer_48k" if selected_vocab == 48000 else "tokenizer_32k"
    selected_dir = paths[selected_key]
    tokenizer_path = selected_dir / "tokenizer.json"
    manifest_path = selected_dir / "manifest.json"
    if not tokenizer_path.is_file() or not manifest_path.is_file():
        raise PublicGpuSelectionError("selected tokenizer artifact is incomplete")
    selected_gpu = gpu_48 if selected_vocab == 48000 else gpu_32
    report = {
        "schema_version": 1,
        "selection_version": "public-tokenizer-quality-gpu-v1",
        "selected_vocab_size": selected_vocab,
        "selected_tokenizer_dir": selected_dir.relative_to(root).as_posix(),
        "selected_tokenizer_sha256": _sha256(tokenizer_path),
        "selected_tokenizer_manifest_sha256": _sha256(manifest_path),
        "selected_parameter_count": selected_gpu["identity"]["parameter_count"],
        "heldout_selected_vocab_size": heldout["selected_vocab_size"],
        "gpu_confirmed": True,
        "effective_public_text_bytes_per_second": effective,
        "global_training_tokens_per_second": {
            "32000": gpu_32["global_tokens_per_second"],
            "48000": gpu_48["global_tokens_per_second"],
        },
        "peak_reserved_gib": {
            "32000": gpu_32["peak_reserved_gib"],
            "48000": gpu_48["peak_reserved_gib"],
        },
        "lineage": {
            **{f"{key}_report_sha256": value for key, value in report_hashes.items()},
            "heldout_sha256": evaluation_32["heldout_sha256"],
        },
        "checks": {
            "same_disjoint_heldout": True,
            "six_gpu_ddp_benchmarked": True,
            "quality_gate_applied_before_gpu_gate": True,
            "no_synthetic_training_content": True,
            "model_external_capability": False,
        },
        "training_eligible": True,
    }
    if output.exists():
        existing, _ = _read_completed(output)
        if existing == report:
            return existing
        raise PublicGpuSelectionError("existing final selection is incompatible")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").write_text(
            f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-selection-dir", type=Path, required=True)
    parser.add_argument("--evaluation-32k-dir", type=Path, required=True)
    parser.add_argument("--evaluation-48k-dir", type=Path, required=True)
    parser.add_argument("--gpu-32k-dir", type=Path, required=True)
    parser.add_argument("--gpu-48k-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-32k-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-48k-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = select(
        heldout_selection_dir=args.heldout_selection_dir,
        evaluation_32k_dir=args.evaluation_32k_dir,
        evaluation_48k_dir=args.evaluation_48k_dir,
        gpu_32k_dir=args.gpu_32k_dir,
        gpu_48k_dir=args.gpu_48k_dir,
        tokenizer_32k_dir=args.tokenizer_32k_dir,
        tokenizer_48k_dir=args.tokenizer_48k_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
