"""Evaluate deterministic generalization prompts using raw checkpoint generation."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def evaluate_responses(responses: dict[str, str]) -> dict[str, object]:
    normalized = {name: value.strip() for name, value in responses.items()}
    checks = {
        "capital_crosslingual": "渥太华" in normalized["capital_crosslingual"],
        "fact_gold": normalized["fact_gold"].casefold() == "au",
        "fact_author": any(
            expected in normalized["fact_author"].casefold()
            for expected in ("george orwell", "乔治·奥威尔", "乔治奥威尔")
        ),
        "instruction_exact": normalized["instruction_exact"] == "OK",
        "code_unseen": (
            "def square" in normalized["code_unseen"]
            and any(
                expression in normalized["code_unseen"]
                for expression in ("return n ** 2", "return n * n")
            )
        ),
        "reasoning_simple": normalized["reasoning_simple"] in {"7", "7。"},
        "comparison": normalized["comparison"] in {"0.8", "0.8。"},
        "unseen_memory": (
            "小华" in normalized["unseen_memory_turn_2"]
            and "橙" in normalized["unseen_memory_turn_2"]
        ),
    }
    return {
        "passed": checks,
        "passed_count": sum(checks.values()),
        "task_count": len(checks),
        "accuracy": sum(checks.values()) / len(checks),
    }


def run(
    checkpoint: Path,
    model_config: Path,
    tokenizer_path: Path,
) -> dict[str, object]:
    manifest_path = checkpoint / "manifest.json"
    model_path = checkpoint / "model.safetensors"
    if not (checkpoint / "COMPLETE").is_file() or not manifest_path.is_file():
        raise ValueError("checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("files", {}).get("model.safetensors")
    if (
        not isinstance(record, dict)
        or model_path.stat().st_size != record.get("bytes")
        or _sha256(model_path) != record.get("sha256")
    ):
        raise ValueError("checkpoint model payload does not match manifest")
    model = AtomLLM(load_model_config(model_config)).to(
        device="cuda", dtype=torch.bfloat16
    )
    load_safetensors_checkpoint(model, model_path)
    model.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def ask(messages: list[dict[str, str]]) -> str:
        return answer(
            model,
            tokenizer,
            messages,
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.9,
            top_k=1,
            seed=42,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )

    prompts = {
        "capital_crosslingual": "加拿大的首都是哪里？只回答城市名。",
        "fact_gold": "黄金的化学元素符号是什么？只回答符号。",
        "fact_author": "Who wrote the novel 1984? Answer with the author name only.",
        "instruction_exact": "只输出字符串 OK，不要输出其他内容。",
        "code_unseen": "Write a Python function square(n) that returns n squared.",
        "reasoning_simple": "小王有12个苹果，送出5个，还剩几个？只回答数字。",
        "comparison": "Which is larger: 0.8 or 0.75? Answer with one number.",
    }
    responses = {
        name: ask([{"role": "user", "content": prompt}])
        for name, prompt in prompts.items()
    }
    first = ask([{"role": "user", "content": "请记住：我叫小华，最喜欢橙色。"}])
    second = ask(
        [
            {"role": "user", "content": "请记住：我叫小华，最喜欢橙色。"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "我叫什么，最喜欢什么颜色？"},
        ]
    )
    responses["unseen_memory_turn_1"] = first
    responses["unseen_memory_turn_2"] = second
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "model_sha256": record["sha256"],
        "summary": evaluate_responses(responses),
        "responses": responses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    report = run(args.checkpoint, args.model_config, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
