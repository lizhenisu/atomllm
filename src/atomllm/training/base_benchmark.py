"""Build and score a frozen public zero-shot benchmark for base checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from datasets import load_dataset
from tokenizers import Tokenizer

from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.config import DistributedConfig
from atomllm.training.distributed import DistributedContext


SOURCE_CONTRACTS = {
    "arc_challenge": {
        "repository": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "config": "ARC-Challenge",
        "split": "test",
    },
    "hellaswag": {
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "config": None,
        "split": "validation",
    },
    "mmlu": {
        "repository": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "config": "all",
        "split": "test",
    },
    "ceval": {
        "repository": "ceval/ceval-exam",
        "revision": "617524a00b307ff6f9933702f724131fe12ca7ce",
        "config": "all-subjects",
        "split": "val",
    },
}
CEVAL_SUBJECTS = (
    "accountant",
    "advanced_mathematics",
    "art_studies",
    "basic_medicine",
    "business_administration",
    "chinese_language_and_literature",
    "civil_servant",
    "clinical_medicine",
    "college_chemistry",
    "college_economics",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "education_science",
    "electrical_engineer",
    "environmental_impact_assessment_engineer",
    "fire_engineer",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_chinese",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "ideological_and_moral_cultivation",
    "law",
    "legal_professional",
    "logic",
    "mao_zedong_thought",
    "marxism",
    "metrology_engineer",
    "middle_school_biology",
    "middle_school_chemistry",
    "middle_school_geography",
    "middle_school_history",
    "middle_school_mathematics",
    "middle_school_physics",
    "middle_school_politics",
    "modern_chinese_history",
    "operating_system",
    "physician",
    "plant_protection",
    "probability_and_statistics",
    "professional_tour_guide",
    "sports_science",
    "tax_accountant",
    "teacher_qualification",
    "urban_and_rural_planner",
    "veterinary_medicine",
)


class BaseBenchmarkError(RuntimeError):
    """Raised when the frozen benchmark contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sample(
    rows: Iterable[dict[str, Any]], count: int, *, seed: int, key: str
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row[key]}".encode("utf-8")).digest(),
    )
    return ranked[:count]


def _task(
    benchmark: str,
    task_id: str,
    prompt: str,
    choices: list[str],
    answer: int,
    *,
    subject: str | None = None,
) -> dict[str, Any]:
    if not 2 <= len(choices) <= 6 or not 0 <= answer < len(choices):
        raise BaseBenchmarkError(f"invalid public task: {benchmark}:{task_id}")
    return {
        "benchmark": benchmark,
        "task_id": f"{benchmark}:{task_id}",
        "subject": subject,
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
    }


def _load_public_tasks(seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    arc_contract = SOURCE_CONTRACTS["arc_challenge"]
    arc = load_dataset(
        arc_contract["repository"],
        arc_contract["config"],
        split=arc_contract["split"],
        revision=arc_contract["revision"],
    )
    for row in arc:
        labels = list(row["choices"]["label"])
        tasks.append(
            _task(
                "arc_challenge",
                str(row["id"]),
                f"Question: {row['question']}\nAnswer: ",
                list(row["choices"]["text"]),
                labels.index(row["answerKey"]),
            )
        )

    hellaswag_contract = SOURCE_CONTRACTS["hellaswag"]
    hellaswag = load_dataset(
        hellaswag_contract["repository"],
        split=hellaswag_contract["split"],
        revision=hellaswag_contract["revision"],
    )
    for row in hellaswag:
        stable_id = (
            f"{row['source_id']}:{row['ind']}:"
            f"{hashlib.sha256(str(row['ctx']).encode()).hexdigest()}"
        )
        tasks.append(
            _task(
                "hellaswag",
                hashlib.sha256(stable_id.encode()).hexdigest()[:20],
                str(row["ctx"]) + " ",
                list(row["endings"]),
                int(row["label"]),
            )
        )

    mmlu_contract = SOURCE_CONTRACTS["mmlu"]
    mmlu = load_dataset(
        mmlu_contract["repository"],
        mmlu_contract["config"],
        split=mmlu_contract["split"],
        revision=mmlu_contract["revision"],
    )
    mmlu_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(mmlu):
        mmlu_by_subject[str(row["subject"])].append(
            {**row, "stable_id": f"{row['subject']}:{index}:{row['question']}"}
        )
    for subject, rows in sorted(mmlu_by_subject.items()):
        for row in rows:
            tasks.append(
                _task(
                    "mmlu",
                    hashlib.sha256(row["stable_id"].encode()).hexdigest()[:20],
                    f"Question: {row['question']}\nAnswer: ",
                    list(row["choices"]),
                    int(row["answer"]),
                    subject=subject,
                )
            )

    ceval_contract = SOURCE_CONTRACTS["ceval"]
    for subject in CEVAL_SUBJECTS:
        dataset = load_dataset(
            ceval_contract["repository"],
            subject,
            split=ceval_contract["split"],
            revision=ceval_contract["revision"],
        )
        rows = [
            {**row, "stable_id": f"{subject}:{row['id']}:{row['question']}"}
            for row in dataset
        ]
        for row in rows:
            choices = [str(row[label]) for label in "ABCD"]
            tasks.append(
                _task(
                    "ceval",
                    hashlib.sha256(row["stable_id"].encode()).hexdigest()[:20],
                    f"问题：{row['question']}\n答案：",
                    choices,
                    "ABCD".index(row["answer"]),
                    subject=subject,
                )
            )
    return tasks


def build_suite(output_dir: Path, *, seed: int) -> dict[str, Any]:
    if output_dir.exists():
        raise BaseBenchmarkError(f"benchmark output already exists: {output_dir}")
    tasks = _load_public_tasks(seed)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise BaseBenchmarkError("public benchmark task IDs are not unique")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        task_path = temporary / "tasks.jsonl"
        with task_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(
                    json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n"
                )
        counts = Counter(task["benchmark"] for task in tasks)
        manifest = {
            "schema_version": 1,
            "suite_id": "atom-base-public-zero-shot-full-v3",
            "created_at": datetime.now(UTC).isoformat(),
            "selection_seed": seed,
            "source_contracts": SOURCE_CONTRACTS,
            "selection": {
                "arc_challenge": "all test rows",
                "hellaswag": "all validation rows",
                "mmlu": "all test rows in every subject",
                "ceval": "all validation rows in every subject",
            },
            "scoring": "raw and per-token-normalized candidate continuation likelihood",
            "model_external_answering": False,
            "task_count": len(tasks),
            "benchmark_counts": dict(sorted(counts.items())),
            "files": {
                "tasks.jsonl": {
                    "bytes": task_path.stat().st_size,
                    "sha256": _sha256(task_path),
                }
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETED").write_text(
            f"manifest_sha256={_sha256(manifest_path)}\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_suite(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = directory / "manifest.json"
    completed = directory / "COMPLETED"
    task_path = directory / "tasks.jsonl"
    if (
        not manifest_path.is_file()
        or not completed.is_file()
        or not task_path.is_file()
    ):
        raise BaseBenchmarkError("public benchmark suite is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completed.read_text(encoding="utf-8") != (
        f"manifest_sha256={_sha256(manifest_path)}\n"
    ):
        raise BaseBenchmarkError("public benchmark manifest hash does not match")
    record = manifest["files"]["tasks.jsonl"]
    if (
        task_path.stat().st_size != record["bytes"]
        or _sha256(task_path) != record["sha256"]
    ):
        raise BaseBenchmarkError("public benchmark task payload does not match")
    tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line]
    if len(tasks) != manifest["task_count"]:
        raise BaseBenchmarkError("public benchmark task count does not match")
    return manifest, tasks


def _verify_checkpoint(directory: Path) -> str:
    manifest_path = directory / "manifest.json"
    model_path = directory / "model.safetensors"
    if not manifest_path.is_file() or not model_path.is_file():
        raise BaseBenchmarkError("checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {}).get("model.safetensors", {}).get("sha256")
    actual = _sha256(model_path)
    if expected != actual:
        raise BaseBenchmarkError("checkpoint model hash does not match")
    return actual


@torch.inference_mode()
def _score_choices(
    model: AtomLLM,
    tokenizer: Tokenizer,
    prompt: str,
    choices: list[str],
) -> list[dict[str, float | int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    rows: list[list[int]] = []
    target_starts: list[int] = []
    for choice in choices:
        candidate_ids = tokenizer.encode(choice, add_special_tokens=False).ids
        if not candidate_ids:
            raise BaseBenchmarkError("candidate tokenized to an empty sequence")
        target_starts.append(1 + len(prompt_ids))
        rows.append([2, *prompt_ids, *candidate_ids])
    length = max(map(len, rows))
    if length > model.max_sequence_length:
        raise BaseBenchmarkError("public benchmark task exceeds model context")
    device = next(model.parameters()).device
    input_ids = torch.full(
        (len(rows), length), model.pad_token_id, dtype=torch.int64, device=device
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, device=device)
        attention_mask[index, : len(row)] = True
        labels[index, target_starts[index] : len(row)] = input_ids[
            index, target_starts[index] : len(row)
        ]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, attention_mask=attention_mask)
    if output.logits is None:
        raise BaseBenchmarkError("base model did not return logits")
    losses = functional.cross_entropy(
        output.logits[:, :-1].transpose(1, 2),
        labels[:, 1:],
        ignore_index=-100,
        reduction="none",
    )
    valid = labels[:, 1:] != -100
    sums = (losses * valid).sum(dim=1)
    counts = valid.sum(dim=1)
    return [
        {
            "token_count": int(counts[index]),
            "log_likelihood": -float(sums[index]),
            "mean_log_likelihood": -float(sums[index] / counts[index]),
        }
        for index in range(len(rows))
    ]


def evaluate(
    checkpoint: Path,
    suite: Path,
    model_config: Path,
    tokenizer_path: Path,
    *,
    distributed: DistributedContext | None = None,
) -> dict[str, Any] | None:
    owns_distributed = distributed is None
    if distributed is None:
        distributed = DistributedContext.initialize(
            DistributedConfig(
                enabled=int(os.environ.get("WORLD_SIZE", "1")) > 1,
                backend="nccl",
            )
        )
    try:
        payload: dict[str, Any] | None = None
        if distributed.is_main_process:
            try:
                manifest, tasks = _load_suite(suite)
                payload = {
                    "ok": True,
                    "manifest": manifest,
                    "tasks": tasks,
                    "model_sha256": _verify_checkpoint(checkpoint),
                    "model_config_sha256": _sha256(model_config),
                    "tokenizer_sha256": _sha256(tokenizer_path),
                }
            except BaseException as error:
                payload = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        payload = distributed.broadcast_object(payload)
        if not payload["ok"]:
            raise BaseBenchmarkError(
                f"rank-0 benchmark verification failed: {payload['error']}"
            )
        manifest = payload["manifest"]
        tasks = payload["tasks"]
        model_sha = payload["model_sha256"]
        world_size = distributed.world_size
        rank = distributed.rank
        device = distributed.device("cuda")
        model = AtomLLM(load_model_config(model_config)).to(
            device=device, dtype=torch.bfloat16
        )
        load_safetensors_checkpoint(model, checkpoint / "model.safetensors")
        model.eval()
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        results = []
        local_tasks = [
            (index, task)
            for index, task in enumerate(tasks)
            if index % world_size == rank
        ]
        for completed, (task_index, task) in enumerate(local_tasks, start=1):
            scores = _score_choices(model, tokenizer, task["prompt"], task["choices"])
            raw_prediction = max(
                range(len(scores)),
                key=lambda choice: scores[choice]["log_likelihood"],
            )
            normalized_prediction = max(
                range(len(scores)),
                key=lambda choice: scores[choice]["mean_log_likelihood"],
            )
            results.append(
                {
                    "task_index": task_index,
                    "benchmark": task["benchmark"],
                    "task_id": task["task_id"],
                    "subject": task["subject"],
                    "answer": task["answer"],
                    "raw_prediction": raw_prediction,
                    "normalized_prediction": normalized_prediction,
                    "raw_passed": raw_prediction == task["answer"],
                    "normalized_passed": normalized_prediction == task["answer"],
                    "choice_count": len(task["choices"]),
                    "scores": scores,
                }
            )
            if completed % 100 == 0:
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "completed": completed,
                            "local_total": len(local_tasks),
                        }
                    ),
                    flush=True,
                )
        if distributed.is_distributed:
            gathered: list[list[dict[str, Any]] | None] | None = (
                [None] * world_size if rank == 0 else None
            )
            dist.gather_object(results, gathered, dst=0)
            distributed.barrier()
            if rank != 0:
                return None
            assert gathered is not None and all(item is not None for item in gathered)
            results = [row for item in gathered if item is not None for row in item]
    finally:
        if owns_distributed:
            distributed.close()
    results.sort(key=lambda row: row["task_index"])
    if [row["task_index"] for row in results] != list(range(len(tasks))):
        raise BaseBenchmarkError("distributed benchmark task coverage is incomplete")
    for row in results:
        del row["task_index"]
    benchmark_metrics = {}
    for benchmark in sorted({result["benchmark"] for result in results}):
        subset = [result for result in results if result["benchmark"] == benchmark]
        benchmark_metrics[benchmark] = {
            "task_count": len(subset),
            "chance_accuracy": sum(1 / row["choice_count"] for row in subset)
            / len(subset),
            "raw_accuracy": sum(row["raw_passed"] for row in subset) / len(subset),
            "normalized_accuracy": sum(row["normalized_passed"] for row in subset)
            / len(subset),
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "model_sha256": model_sha,
        "model_config_sha256": payload["model_config_sha256"],
        "tokenizer_sha256": payload["tokenizer_sha256"],
        "suite_manifest_sha256": _sha256(suite / "manifest.json"),
        "suite_id": manifest["suite_id"],
        "model_external_answering": False,
        "world_size": world_size,
        "task_count": len(results),
        "raw_accuracy": sum(row["raw_passed"] for row in results) / len(results),
        "normalized_accuracy": sum(row["normalized_passed"] for row in results)
        / len(results),
        "benchmark_metrics": benchmark_metrics,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-suite", action="store_true")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/atom-base-300m-long-v1.yaml"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/atom-tokenizer-formal-v4/tokenizer.json"),
    )
    args = parser.parse_args(argv)
    if args.build_suite:
        if args.checkpoint is not None or args.output is not None:
            raise BaseBenchmarkError("suite building does not accept checkpoint/output")
        report = build_suite(args.suite, seed=args.seed)
    else:
        if args.checkpoint is None or args.output is None:
            raise BaseBenchmarkError("evaluation requires checkpoint and output")
        report = evaluate(
            args.checkpoint, args.suite, args.model_config, args.tokenizer
        )
        if report is None:
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
