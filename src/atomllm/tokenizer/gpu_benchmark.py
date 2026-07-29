"""Benchmark one public tokenizer with the real 300M DDP training workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from atomllm.data.schema import CanonicalDocument
from atomllm.model.config import calculate_parameter_count, load_model_config
from atomllm.model.model import AtomLLM
from atomllm.tokenizer.evaluation import verify_tokenizer_directory


class PublicTokenizerGpuBenchmarkError(RuntimeError):
    """Raised when a tokenizer GPU benchmark is invalid or incomplete."""


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
        raise PublicTokenizerGpuBenchmarkError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PublicTokenizerGpuBenchmarkError(f"JSON must be an object: {path}")
    return value


def _resolve(root: Path, path: Path, field: str) -> Path:
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not result.is_relative_to(root):
        raise PublicTokenizerGpuBenchmarkError(f"{field} escapes project root")
    return result


def _verified_evaluation(
    directory: Path,
    *,
    tokenizer_manifest_sha256: str,
    vocab_size: int,
) -> tuple[dict[str, Any], str]:
    report_path = directory / "report.json"
    completed_path = directory / "COMPLETED"
    if not report_path.is_file() or not completed_path.is_file():
        raise PublicTokenizerGpuBenchmarkError("held-out evaluation is incomplete")
    report_sha = _sha256(report_path)
    if completed_path.read_text(encoding="utf-8") != f"{report_sha}  report.json\n":
        raise PublicTokenizerGpuBenchmarkError("held-out evaluation marker is invalid")
    report = _read_json(report_path)
    if report.get("tokenizer_manifest_sha256") != tokenizer_manifest_sha256:
        raise PublicTokenizerGpuBenchmarkError("evaluation tokenizer lineage mismatch")
    if report.get("vocab_size") != vocab_size:
        raise PublicTokenizerGpuBenchmarkError("evaluation vocabulary mismatch")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise PublicTokenizerGpuBenchmarkError("evaluation summary is invalid")
    if summary.get("unknown_count") != 0 or summary.get("roundtrip_failures") != 0:
        raise PublicTokenizerGpuBenchmarkError("tokenizer correctness gate failed")
    return report, report_sha


def _packed_public_tokens(
    heldout_path: Path,
    tokenizer: Any,
    *,
    required_tokens: int,
) -> torch.Tensor:
    values: list[int] = []
    with heldout_path.open(encoding="utf-8") as handle:
        for line in handle:
            document = CanonicalDocument.from_json_line(line)
            encoding = tokenizer.encode(document.text, add_special_tokens=False)
            if 1 in encoding.ids:
                raise PublicTokenizerGpuBenchmarkError(
                    "tokenizer emitted <unk> on benchmark data"
                )
            values.extend((2, *encoding.ids, 3))
            if len(values) >= required_tokens:
                break
    if not values:
        raise PublicTokenizerGpuBenchmarkError("benchmark held-out data is empty")
    original = tuple(values)
    while len(values) < required_tokens:
        missing = required_tokens - len(values)
        values.extend(original[:missing])
    return torch.tensor(values[:required_tokens], dtype=torch.int64)


def _candidate_model(base_path: Path, manifest: dict[str, Any], tokenizer_sha: str):
    base = load_model_config(base_path)
    vocab_size = manifest.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise PublicTokenizerGpuBenchmarkError("tokenizer vocabulary is invalid")
    tokenizer = replace(
        base.tokenizer,
        version_id=f"tokenizer-version-public-benchmark-{tokenizer_sha[:12]}",
        tokenizer_sha256=tokenizer_sha,
        vocab_size=vocab_size,
    )
    provisional = replace(base, tokenizer=tokenizer)
    return replace(
        provisional,
        name=f"atom-base-public-{vocab_size // 1000}k-benchmark",
        status="smoke",
        expected_parameter_count=calculate_parameter_count(provisional).total,
    )


def benchmark(
    *,
    tokenizer_dir: Path,
    evaluation_dir: Path,
    snapshot_dir: Path,
    base_model_config: Path,
    output_dir: Path,
    sequence_length: int = 4096,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 2,
    measured_steps: int = 5,
    loss_chunk_size: int = 1024,
    ddp_bucket_cap_mb: int = 200,
    expected_world_size: int = 6,
    seed: int = 20260719,
    project_root: Path = Path("."),
) -> dict[str, Any] | None:
    positive = {
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "loss_chunk_size": loss_chunk_size,
        "ddp_bucket_cap_mb": ddp_bucket_cap_mb,
        "expected_world_size": expected_world_size,
    }
    if any(type(value) is not int or value <= 0 for value in positive.values()):
        raise PublicTokenizerGpuBenchmarkError(
            "benchmark sizes must be positive integers"
        )
    if not torch.cuda.is_available():
        raise PublicTokenizerGpuBenchmarkError("CUDA is required")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != expected_world_size or not 0 <= rank < world_size:
        raise PublicTokenizerGpuBenchmarkError(
            f"torchrun world size must be exactly {expected_world_size}"
        )
    if not 0 <= local_rank < torch.cuda.device_count():
        raise PublicTokenizerGpuBenchmarkError("LOCAL_RANK is invalid")
    root = project_root.resolve()
    tokenizer_path = _resolve(root, tokenizer_dir, "tokenizer_dir")
    evaluation_path = _resolve(root, evaluation_dir, "evaluation_dir")
    snapshot_path = _resolve(root, snapshot_dir, "snapshot_dir")
    model_path = _resolve(root, base_model_config, "base_model_config")
    output_path = _resolve(root, output_dir, "output_dir")
    tokenizer, tokenizer_manifest, tokenizer_manifest_path = verify_tokenizer_directory(
        tokenizer_path
    )
    if tokenizer_manifest.get("training_eligible") is not True:
        raise PublicTokenizerGpuBenchmarkError("tokenizer is not training eligible")
    for token, expected_id in (("<pad>", 0), ("<unk>", 1), ("<bos>", 2), ("<eos>", 3)):
        if tokenizer.token_to_id(token) != expected_id:
            raise PublicTokenizerGpuBenchmarkError(
                f"tokenizer special-token ID mismatch: {token}"
            )
    tokenizer_sha = _sha256(tokenizer_path / "tokenizer.json")
    tokenizer_manifest_sha = _sha256(tokenizer_manifest_path)
    evaluation, evaluation_sha = _verified_evaluation(
        evaluation_path,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
        vocab_size=tokenizer_manifest["vocab_size"],
    )
    heldout_path = snapshot_path / "heldout.jsonl"
    if not heldout_path.is_file() or _sha256(heldout_path) != evaluation.get(
        "heldout_sha256"
    ):
        raise PublicTokenizerGpuBenchmarkError("benchmark held-out lineage mismatch")
    model_config = _candidate_model(model_path, tokenizer_manifest, tokenizer_sha)
    identity = {
        "benchmark_version": "public-tokenizer-gpu-ddp-v1",
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
        "tokenizer_sha256": tokenizer_sha,
        "evaluation_sha256": evaluation_sha,
        "heldout_sha256": evaluation["heldout_sha256"],
        "base_model_config_sha256": _sha256(model_path),
        "vocab_size": model_config.tokenizer.vocab_size,
        "parameter_count": model_config.expected_parameter_count,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "loss_chunk_size": loss_chunk_size,
        "ddp_bucket_cap_mb": ddp_bucket_cap_mb,
        "world_size": world_size,
        "precision": "bf16",
        "gradient_checkpointing": False,
        "seed": seed,
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        skip = False
        existing: dict[str, Any] | None = None
        if rank == 0 and output_path.exists():
            report_path = output_path / "report.json"
            completed_path = output_path / "COMPLETED"
            if report_path.is_file() and completed_path.is_file():
                existing = _read_json(report_path)
                skip = existing.get("identity_sha256") == identity_sha and (
                    completed_path.read_text(encoding="utf-8")
                    == f"{_sha256(report_path)}  report.json\n"
                )
            if not skip:
                raise PublicTokenizerGpuBenchmarkError(
                    "existing GPU benchmark is incompatible"
                )
        flag = torch.tensor(int(skip), dtype=torch.int32, device=device)
        dist.broadcast(flag, src=0)
        if int(flag.item()):
            return existing if rank == 0 else None

        total_steps = warmup_steps + measured_steps
        tokens_per_rank_step = (
            micro_batch_size * gradient_accumulation_steps * sequence_length
        )
        required_tokens = total_steps * tokens_per_rank_step * world_size
        global_tokens = torch.empty(required_tokens, dtype=torch.int64, device=device)
        if rank == 0:
            cpu_tokens = _packed_public_tokens(
                heldout_path,
                tokenizer,
                required_tokens=required_tokens,
            )
            global_tokens.copy_(cpu_tokens, non_blocking=False)
        dist.broadcast(global_tokens, src=0)
        batches = global_tokens.view(
            total_steps,
            gradient_accumulation_steps,
            world_size,
            micro_batch_size,
            sequence_length,
        )[:, :, rank].contiguous()
        del global_tokens

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = AtomLLM(model_config).to(device=device, dtype=torch.float32)
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=ddp_bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-4,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
            fused=True,
        )

        measured_loss = 0.0
        measured_gradient_norm = 0.0
        started = 0.0
        for step in range(total_steps):
            if step == warmup_steps:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                dist.barrier()
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro_step in range(gradient_accumulation_steps):
                sync = micro_step == gradient_accumulation_steps - 1
                context = nullcontext() if sync else training_model.no_sync()
                with context, torch.autocast("cuda", dtype=torch.bfloat16):
                    batch = batches[step, micro_step]
                    output = training_model(
                        batch,
                        labels=batch,
                        loss_chunk_size=loss_chunk_size,
                        gradient_checkpointing=False,
                    )
                    if output.loss is None:
                        raise PublicTokenizerGpuBenchmarkError("model returned no loss")
                    (output.loss / gradient_accumulation_steps).backward()
                    step_loss += float(output.loss.detach())
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            )
            optimizer.step()
            if step >= warmup_steps:
                measured_loss += step_loss / gradient_accumulation_steps
                measured_gradient_norm += gradient_norm
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        allocated = torch.tensor(
            torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device
        )
        reserved = torch.tensor(
            torch.cuda.max_memory_reserved(device), dtype=torch.float64, device=device
        )
        dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
        loss_tensor = torch.tensor(measured_loss, dtype=torch.float64, device=device)
        norm_tensor = torch.tensor(
            measured_gradient_norm, dtype=torch.float64, device=device
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
        if rank != 0:
            return None
        measured_global_tokens = measured_steps * tokens_per_rank_step * world_size
        report = {
            "schema_version": 1,
            "identity_sha256": identity_sha,
            "identity": identity,
            "elapsed_seconds": float(elapsed),
            "global_training_tokens": measured_global_tokens,
            "global_tokens_per_second": measured_global_tokens / float(elapsed),
            "mean_loss": float(loss_tensor) / (measured_steps * world_size),
            "mean_gradient_norm": float(norm_tensor) / (measured_steps * world_size),
            "peak_allocated_gib": float(allocated) / 1024**3,
            "peak_reserved_gib": float(reserved) / 1024**3,
            "gpu_name": torch.cuda.get_device_name(device),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(str(item) for item in torch.cuda.nccl.version()),
            },
            "checks": {
                "six_gpu_ddp": world_size == 6,
                "real_public_heldout_tokens": True,
                "weights_discarded_after_benchmark": True,
                "model_external_capability": False,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
        )
        try:
            report_path = temporary / "report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (temporary / "COMPLETED").write_text(
                f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
            )
            os.replace(temporary, output_path)
        except BaseException:
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return report
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-config",
        type=Path,
        default=Path("configs/model/atom-base-300m.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--loss-chunk-size", type=int, default=1024)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=200)
    parser.add_argument("--expected-world-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = benchmark(
        tokenizer_dir=args.tokenizer_dir,
        evaluation_dir=args.evaluation_dir,
        snapshot_dir=args.snapshot_dir,
        base_model_config=args.base_model_config,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        loss_chunk_size=args.loss_chunk_size,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        expected_world_size=args.expected_world_size,
        seed=args.seed,
        project_root=args.project_root,
    )
    if report is not None:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
