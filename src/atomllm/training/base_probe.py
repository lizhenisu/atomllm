"""Score fixed factual and arithmetic continuations for a base checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer

from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.post_training.sft_training import _verify_base_checkpoint


PROBES = (
    ("capital_zh", "中国的首都是", ("北京", "上海", "广州", "深圳"), "北京"),
    ("addition_seen", "2 + 3 =", ("5", "4", "6", "3"), "5"),
    ("addition_generalization", "17 + 28 =", ("45", "44", "46", "35"), "45"),
    ("multiplication", "9 × 7 =", ("63", "56", "72", "12"), "63"),
    (
        "photosynthesis_en",
        "Photosynthesis allows plants to use",
        (" sunlight", " sound", " gravity", " plastic"),
        " sunlight",
    ),
    (
        "python_add",
        "def add(a, b):\n    return",
        (" a + b", " sum(a)", " b", " None"),
        " a + b",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def score_candidate(
    model: AtomLLM, tokenizer: Tokenizer, prompt: str, candidate: str
) -> dict[str, float | int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False).ids
    tokens = [2, *prompt_ids, *candidate_ids]
    labels = [-100] * (1 + len(prompt_ids)) + candidate_ids
    device = next(model.parameters()).device
    input_tensor = torch.tensor([tokens], dtype=torch.int64, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.int64, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_tensor, labels=label_tensor)
    if output.loss is None or not torch.isfinite(output.loss).item():
        raise RuntimeError("probe candidate loss is not finite")
    mean_log_probability = -float(output.loss)
    return {
        "token_count": len(candidate_ids),
        "mean_log_probability": mean_log_probability,
        "per_token_probability": math.exp(mean_log_probability),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    _verify_base_checkpoint(
        args.checkpoint,
        expected_manifest_sha256=args.manifest_sha256,
        expected_model_sha256=args.model_sha256,
    )
    model = AtomLLM(load_model_config(args.model_config)).to(
        device="cuda", dtype=torch.float32
    )
    load_safetensors_checkpoint(model, args.checkpoint / "model.safetensors")
    model.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    results = []
    correct_at_1 = 0
    reciprocal_rank = 0.0
    for name, prompt, candidates, expected in PROBES:
        scores = [
            {
                "candidate": candidate,
                **score_candidate(model, tokenizer, prompt, candidate),
            }
            for candidate in candidates
        ]
        scores.sort(key=lambda item: item["mean_log_probability"], reverse=True)
        rank = next(
            index
            for index, item in enumerate(scores, start=1)
            if item["candidate"] == expected
        )
        correct_at_1 += rank == 1
        reciprocal_rank += 1 / rank
        results.append(
            {
                "name": name,
                "prompt": prompt,
                "expected": expected,
                "expected_rank": rank,
                "scores": scores,
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_manifest_sha256": args.manifest_sha256,
        "model_sha256": args.model_sha256,
        "model_config_sha256": _sha256(args.model_config),
        "tokenizer_sha256": _sha256(args.tokenizer),
        "probe_count": len(PROBES),
        "accuracy_at_1": correct_at_1 / len(PROBES),
        "mean_reciprocal_rank": reciprocal_rank / len(PROBES),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
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
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
