"""Evaluate generated short answers on deterministic basic arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer

from atomllm.inference.chat import answer
from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tasks() -> list[tuple[str, str, str]]:
    """Legacy tasks retained only for comparison with historical reports."""
    tasks: list[tuple[str, str, str]] = []
    for index in range(25):
        left = (index * 17 + 3) % 100
        right = (index * 29 + 7) % 100
        tasks.append(
            ("add", f"直接计算 {left} 加 {right}，只输出整数。", str(left + right))
        )

        high = (index * 19 + 41) % 141
        low = (index * 11 + 5) % (high + 1)
        tasks.append(
            (
                "subtract",
                f"Answer with one integer: {high} - {low} =",
                str(high - low),
            )
        )

        factor_a = (index * 7 + 2) % 21
        factor_b = (index * 13 + 3) % 21
        tasks.append(
            (
                "multiply",
                f"直接计算 {factor_a} 乘 {factor_b}，只输出整数。",
                str(factor_a * factor_b),
            )
        )

        divisor = index % 20 + 1
        quotient = (index * 9 + 4) % 21
        dividend = divisor * quotient
        tasks.append(
            (
                "divide",
                f"Answer with one integer: {dividend} / {divisor} =",
                str(quotient),
            )
        )
    return tasks


def _heldout_tasks() -> list[tuple[str, str, str, str]]:
    """Return template- and range-held-out arithmetic generation tasks."""
    tasks: list[tuple[str, str, str, str]] = []
    for index in range(25):
        left = (index * 17 + 3) % 100
        right = (index * 29 + 7) % 100
        tasks.append(
            (
                "template_holdout",
                "add",
                f"不要展示过程。把 {left} 和 {right} 相加后，仅写最终整数。",
                str(left + right),
            )
        )
        high = (index * 19 + 41) % 141
        low = (index * 11 + 5) % (high + 1)
        tasks.append(
            (
                "template_holdout",
                "subtract",
                f"No working: take {low} away from {high}; reply with the integer left.",
                str(high - low),
            )
        )
        factor_a = index % 21
        factor_b = (index * 13 + index // 21) % 21
        tasks.append(
            (
                "template_holdout",
                "multiply",
                f"不要列步骤。{factor_a} 的 {factor_b} 倍是多少？只写整数。",
                str(factor_a * factor_b),
            )
        )
        divisor = index % 20 + 1
        quotient = (index * 9 + 4) % 21
        dividend = divisor * quotient
        tasks.append(
            (
                "template_holdout",
                "divide",
                f"No explanation: split {dividend} into groups of {divisor}; give the group count.",
                str(quotient),
            )
        )

        large_left = 101 + (index * 37) % 899
        large_right = 103 + (index * 53) % 897
        tasks.append(
            (
                "range_holdout",
                "add",
                f"Without showing work, state the sum of {large_left} and {large_right} as one integer.",
                str(large_left + large_right),
            )
        )
        large_high = 301 + (index * 61) % 699
        large_low = 101 + (index * 43) % 199
        tasks.append(
            (
                "range_holdout",
                "subtract",
                f"不要解释：从 {large_high} 中拿走 {large_low}，还剩多少？仅写整数。",
                str(large_high - large_low),
            )
        )
        large_factor_a = 21 + (index * 11) % 79
        large_factor_b = 22 + (index * 17) % 78
        tasks.append(
            (
                "range_holdout",
                "multiply",
                f"Give just the integer product when {large_factor_a} is repeated {large_factor_b} times.",
                str(large_factor_a * large_factor_b),
            )
        )
        large_divisor = 21 + (index * 13) % 79
        large_quotient = 22 + (index * 19) % 78
        large_dividend = large_divisor * large_quotient
        tasks.append(
            (
                "range_holdout",
                "divide",
                f"把 {large_dividend} 平均分成每份 {large_divisor}，共有几份？答案只写整数。",
                str(large_quotient),
            )
        )
    return tasks


def _normalized_integer(value: str) -> str | None:
    match = re.fullmatch(r"\s*([+-]?\d+)\s*[。.!！]?\s*", value)
    return None if match is None else str(int(match.group(1)))


def run(
    checkpoint: Path,
    model_config: Path,
    tokenizer_path: Path,
    *,
    suite: str = "heldout-v2",
) -> dict[str, object]:
    if suite not in {"heldout-v2", "legacy"}:
        raise ValueError(f"unsupported arithmetic probe suite: {suite}")
    model_path = checkpoint / "model.safetensors"
    model = AtomLLM(load_model_config(model_config)).to(
        device="cuda", dtype=torch.bfloat16
    )
    load_safetensors_checkpoint(model, model_path)
    model.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    correct: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    split_correct: Counter[str] = Counter()
    split_totals: Counter[str] = Counter()
    results = []
    tasks = (
        [("legacy", *task) for task in _tasks()]
        if suite == "legacy"
        else _heldout_tasks()
    )
    for split, operation, prompt, expected in tasks:
        response = answer(
            model,
            tokenizer,
            [{"role": "user", "content": prompt}],
            max_new_tokens=12,
            temperature=0.2,
            top_p=0.9,
            top_k=1,
            seed=42,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )
        normalized = _normalized_integer(response)
        passed = normalized == expected
        totals[operation] += 1
        correct[operation] += int(passed)
        split_totals[split] += 1
        split_correct[split] += int(passed)
        results.append(
            {
                "split": split,
                "operation": operation,
                "prompt": prompt,
                "expected": expected,
                "response": response,
                "normalized_response": normalized,
                "passed": passed,
            }
        )
    total = sum(totals.values())
    passed = sum(correct.values())
    return {
        "schema_version": 2,
        "suite": suite,
        "generalization_contract": (
            {
                "template_holdout": "training-range operands with unseen prompts",
                "range_holdout": "unseen prompts and operands outside training ranges",
                "model_external_answering": False,
            }
            if suite == "heldout-v2"
            else None
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "model_sha256": _sha256(model_path),
        "task_count": total,
        "accuracy": passed / total,
        "operation_accuracy": {
            name: correct[name] / totals[name] for name in sorted(totals)
        },
        "split_accuracy": {
            name: split_correct[name] / split_totals[name]
            for name in sorted(split_totals)
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("heldout-v2", "legacy"), default="heldout-v2"
    )
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
    report = run(
        args.checkpoint,
        args.model_config,
        args.tokenizer,
        suite=args.suite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "model_sha256": report["model_sha256"],
                "task_count": report["task_count"],
                "accuracy": report["accuracy"],
                "operation_accuracy": report["operation_accuracy"],
                "split_accuracy": report["split_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
