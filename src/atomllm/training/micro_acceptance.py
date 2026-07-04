"""Run the stage-3 micro-model acceptance check on one fixed batch."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atomllm.experiment import set_seed
from atomllm.inference.generation import greedy_generate
from atomllm.model.checkpoint import (
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from atomllm.model.config import calculate_parameter_count, load_model_config
from atomllm.model.model import AtomLLM


@dataclass(frozen=True, slots=True)
class MicroAcceptanceReport:
    model_name: str
    parameter_count: int
    device: str
    dtype: str
    steps: int
    initial_loss: float
    final_loss: float
    minimum_loss: float
    all_losses_finite: bool
    loss_decreased: bool
    checkpoint_logits_exact: bool
    cache_generation_exact: bool
    checkpoint_path: str
    passed: bool


def _fixed_batch(
    vocab_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    pattern = torch.arange(sequence_length, device=device)
    return ((pattern * 37 + 15) % (vocab_size - 15) + 15).unsqueeze(0)


def run_micro_acceptance(
    *,
    config_path: Path,
    output_dir: Path,
    steps: int,
    learning_rate: float,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> MicroAcceptanceReport:
    """Train, checkpoint, restore, and compare generation for the micro model."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")

    set_seed(seed)
    config = load_model_config(config_path)
    if sequence_length > config.dimensions.max_sequence_length:
        raise ValueError("sequence_length exceeds the model context")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AtomLLM(config).to(device=device, dtype=dtype).train()
    batch = _fixed_batch(config.tokenizer.vocab_size, sequence_length, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return a training loss")
        loss_value = float(output.loss.detach())
        losses.append(loss_value)
        if not math.isfinite(loss_value):
            break
        output.loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        reference_logits = model(batch).logits
    checkpoint_path = save_safetensors_checkpoint(
        model,
        output_dir / "atom-micro-4m.safetensors",
    )
    restored = AtomLLM(config).to(device=device, dtype=dtype).eval()
    load_safetensors_checkpoint(restored, checkpoint_path)
    with torch.inference_mode():
        restored_logits = restored(batch).logits

    prompt = batch[:, : min(8, sequence_length)]
    without_cache = greedy_generate(
        restored,
        prompt,
        max_new_tokens=8,
        use_cache=False,
    )
    with_cache = greedy_generate(
        restored,
        prompt,
        max_new_tokens=8,
        use_cache=True,
    )
    all_losses_finite = bool(losses) and all(math.isfinite(loss) for loss in losses)
    loss_decreased = (
        all_losses_finite and losses[-1] < losses[0] * 0.1 and losses[-1] < 1.0
    )
    checkpoint_logits_exact = torch.equal(reference_logits, restored_logits)
    cache_generation_exact = torch.equal(without_cache, with_cache)
    passed = loss_decreased and checkpoint_logits_exact and cache_generation_exact
    return MicroAcceptanceReport(
        model_name=config.name,
        parameter_count=calculate_parameter_count(config).total,
        device=str(device),
        dtype=str(dtype),
        steps=len(losses),
        initial_loss=losses[0],
        final_loss=losses[-1],
        minimum_loss=min(losses),
        all_losses_finite=all_losses_finite,
        loss_decreased=loss_decreased,
        checkpoint_logits_exact=checkpoint_logits_exact,
        cache_generation_exact=cache_generation_exact,
        checkpoint_path=str(checkpoint_path),
        passed=passed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/atom-micro-4m.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model/micro-acceptance"),
    )
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report = run_micro_acceptance(
        config_path=args.config,
        output_dir=args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        seed=args.seed,
        device=torch.device(device_name),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
