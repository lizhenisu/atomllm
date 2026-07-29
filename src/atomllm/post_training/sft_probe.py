"""Run deterministic before/after probes against an SFT checkpoint."""

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

from atomllm.inference.chat import answer, latest_checkpoint
from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM


PROBES = (
    ("capital_zh", "中国的首都是哪里？"),
    ("addition_seen", "2 + 3 等于多少？只给出答案。"),
    ("addition_generalization", "17 + 28 等于多少？只给出答案。"),
    ("multiplication", "9 × 7 等于多少？只给出答案。"),
    ("ml_zh", "用两句话向初学者解释什么是机器学习。"),
    ("photosynthesis_en", "Explain photosynthesis in two concise sentences."),
    ("python", "Write a Python function add(a, b) that returns their sum."),
    ("renewable", "Name three renewable energy sources."),
)

STRICT_EXPECTED = {
    "capital_zh": "北京",
    "addition_seen": "5",
    "addition_generalization": "45",
    "multiplication": "63",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_answer(value: str) -> str:
    return re.sub(r"[\s。.!！]+", "", value)


def _four_gram_repetition(value: str) -> float:
    tokens = re.findall(r"[\w]+|[^\w\s]", value.casefold(), flags=re.UNICODE)
    if len(tokens) < 4:
        return 0.0
    grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values())
    return repeated / len(grams)


def evaluate_responses(responses: dict[str, str]) -> dict[str, object]:
    strict = {
        name: _normalized_answer(responses.get(name, "")) == expected
        for name, expected in STRICT_EXPECTED.items()
    }
    repetition = {
        name: _four_gram_repetition(value) for name, value in responses.items()
    }
    memory = responses.get("memory_turn_2", "")
    return {
        "strict_answer_pass": strict,
        "strict_answer_accuracy": sum(strict.values()) / len(strict),
        "memory_pass": "小明" in memory and "蓝" in memory,
        "all_nonempty": all(value.strip() for value in responses.values()),
        "four_gram_repetition": repetition,
        "max_four_gram_repetition": max(repetition.values(), default=0.0),
        "repetition_gate_pass": max(repetition.values(), default=0.0) <= 0.2,
    }


def run_probe(
    run_dir: Path,
    model_config: Path,
    tokenizer_path: Path,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    checkpoint = latest_checkpoint(run_dir) if checkpoint is None else checkpoint
    if checkpoint.parent.parent.resolve() != run_dir.resolve():
        raise ValueError("explicit checkpoint must belong to run_dir/checkpoints")
    manifest_path = checkpoint / "manifest.json"
    if not (checkpoint / "COMPLETE").is_file() or not manifest_path.is_file():
        raise ValueError("explicit checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_record = manifest.get("files", {}).get("model.safetensors")
    model_path = checkpoint / "model.safetensors"
    if (
        not isinstance(model_record, dict)
        or not model_path.is_file()
        or model_path.stat().st_size != model_record["bytes"]
        or _sha256(model_path) != model_record["sha256"]
    ):
        raise ValueError("explicit checkpoint model payload does not match manifest")
    device = torch.device("cuda")
    model = AtomLLM(load_model_config(model_config)).to(
        device=device, dtype=torch.bfloat16
    )
    load_safetensors_checkpoint(model, checkpoint / "model.safetensors")
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

    responses = {
        name: ask([{"role": "user", "content": prompt}]) for name, prompt in PROBES
    }
    first = ask([{"role": "user", "content": "记住：我叫小明，我最喜欢蓝色。"}])
    second = ask(
        [
            {"role": "user", "content": "记住：我叫小明，我最喜欢蓝色。"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "我叫什么，最喜欢什么颜色？"},
        ]
    )
    responses["memory_turn_1"] = first
    responses["memory_turn_2"] = second
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "model_sha256": _sha256(checkpoint / "model.safetensors"),
        "decoding": {
            "max_new_tokens": 96,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 1,
            "seed": 42,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 4,
        },
        "summary": evaluate_responses(responses),
        "responses": responses,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate one verified checkpoint in run-dir; defaults to latest.",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(
        args.run_dir,
        args.model_config,
        args.tokenizer,
        checkpoint=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
