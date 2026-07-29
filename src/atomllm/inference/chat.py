"""Interactive chat CLI for AtomLLM stage-8 SFT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from atomllm.inference.generation import sample_generate
from atomllm.model.checkpoint import load_safetensors_checkpoint
from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM


BOS_ID = 2
EOS_ID = 3
ROLE_IDS = {"system": 4, "user": 5, "assistant": 6}
EOT_ID = 8
GENERATION_STOP_IDS = frozenset({EOS_ID, EOT_ID, *ROLE_IDS.values()})


class ChatError(RuntimeError):
    """Raised when an SFT chat checkpoint or prompt is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(run_dir: Path) -> Path:
    latest_path = run_dir / "checkpoints" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints" / latest["checkpoint_id"]
    manifest_path = checkpoint / "manifest.json"
    if not (checkpoint / "COMPLETE").is_file():
        raise ChatError(f"checkpoint is incomplete: {checkpoint}")
    if _sha256(manifest_path) != latest["manifest_sha256"]:
        raise ChatError("latest checkpoint manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_record = manifest["files"]["model.safetensors"]
    model_path = checkpoint / "model.safetensors"
    if (
        model_path.stat().st_size != model_record["bytes"]
        or _sha256(model_path) != model_record["sha256"]
    ):
        raise ChatError("checkpoint model.safetensors hash mismatch")
    return checkpoint


def encode_chat(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
) -> list[int]:
    tokens = [BOS_ID]
    for message in messages:
        role = message["role"]
        if role not in ROLE_IDS or not message["content"].strip():
            raise ChatError(
                "chat messages require a supported role and non-empty content"
            )
        tokens.append(ROLE_IDS[role])
        tokens.extend(
            tokenizer.encode(message["content"].strip(), add_special_tokens=False).ids
        )
        tokens.append(EOT_ID)
    tokens.append(ROLE_IDS["assistant"])
    return tokens


def answer(
    model: AtomLLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 4,
) -> str:
    prompt = encode_chat(tokenizer, messages)
    input_ids = torch.tensor(
        [prompt], dtype=torch.int64, device=next(model.parameters()).device
    )
    generated = sample_generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_token_ids=GENERATION_STOP_IDS,
        seed=seed,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    new_tokens = generated[0, len(prompt) :].tolist()
    if new_tokens and new_tokens[-1] in GENERATION_STOP_IDS:
        new_tokens.pop()
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with an AtomLLM SFT checkpoint.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/training-runs/atom-chat-300m-sft-v1"),
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
    parser.add_argument(
        "--prompt", help="Ask one question and exit; omit for interactive mode."
    )
    parser.add_argument("--system")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ChatError("CUDA inference requested but unavailable")
    checkpoint = latest_checkpoint(args.run_dir)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AtomLLM(load_model_config(args.model_config)).to(device=device, dtype=dtype)
    load_safetensors_checkpoint(model, checkpoint / "model.safetensors")
    model.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def ask(prompt: str) -> str:
        messages.append({"role": "user", "content": prompt})
        response = answer(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        messages.append({"role": "assistant", "content": response})
        return response

    print(f"Loaded {checkpoint}")
    if args.prompt is not None:
        print(ask(args.prompt))
        return 0
    print("输入问题开始对话；输入 /exit 退出，/clear 清空上下文。")
    while True:
        try:
            prompt = input("\n你：").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if prompt == "/exit":
            break
        if prompt == "/clear":
            messages[:] = messages[:1] if args.system else []
            print("上下文已清空。")
            continue
        if not prompt:
            continue
        print(f"Atom：{ask(prompt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
