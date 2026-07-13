import os
import socket
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.multiprocessing as mp

from atomllm.model.config import load_model_config
from atomllm.model.model import AtomLLM
from atomllm.training.checkpoint import (
    CheckpointIdentity,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from atomllm.training.config import DistributedConfig, file_sha256, load_training_config
from atomllm.training.data import PackedTokenDataset, ResumableBatchIterator
from atomllm.training.distributed import DistributedContext
from atomllm.training.trainer import Trainer


MODEL_CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/atom-5m-baseline.yaml")


def _tiny_model() -> AtomLLM:
    config = load_model_config(MODEL_CONFIG_PATH)
    config = replace(
        config,
        name="atom-ddp-test",
        tokenizer=replace(config.tokenizer, vocab_size=15),
        dimensions=replace(
            config.dimensions,
            max_sequence_length=4,
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            ffn_hidden_size=32,
        ),
        expected_parameter_count=2_592,
    )
    return AtomLLM(config)


def _tiny_config():
    config = load_training_config(TRAINING_CONFIG_PATH)
    return replace(
        config,
        model=replace(
            config.model, name="atom-ddp-test", expected_parameter_count=2_592
        ),
        batch=replace(
            config.batch,
            sequence_length=4,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
        ),
        optimizer=replace(config.optimizer, learning_rate=0.01, weight_decay=0.0),
        scheduler=replace(
            config.scheduler,
            warmup_steps=1,
            total_steps=4,
            minimum_learning_rate_ratio=0.1,
        ),
        runtime=replace(config.runtime, device="cpu", precision="fp32"),
        distributed=DistributedConfig(enabled=True, backend="gloo"),
    )


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        run_id="ddp-exact-resume-test",
        project_version="0.1.0",
        git_commit="0" * 40,
        git_dirty=False,
        tokenizer_sha256="a" * 64,
        config_sha256=file_sha256(TRAINING_CONFIG_PATH),
    )


def _worker(
    rank: int,
    world_size: int,
    port: int,
    dataset_path: str,
    checkpoint_path: str,
    result_path: str,
    mode: str,
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    context = DistributedContext.initialize(
        DistributedConfig(enabled=True, backend="gloo")
    )
    try:
        torch.manual_seed(7301)
        config = _tiny_config()
        dataset = PackedTokenDataset(dataset_path)
        iterator = ResumableBatchIterator(
            dataset,
            batch_size=config.batch.micro_batch_size,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
        )
        first_batch = iterator.next_batch().clone()
        iterator = ResumableBatchIterator(
            dataset,
            batch_size=config.batch.micro_batch_size,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
        )
        trainer = Trainer(_tiny_model(), config, iterator, context)
        if mode == "continuous":
            trainer.train(4)
        elif mode == "interrupt":
            trainer.train(2)
            save_training_checkpoint(trainer, checkpoint_path, _identity(), keep_last=2)
            return
        elif mode == "resume":
            restore_training_checkpoint(trainer, checkpoint_path, _identity())
            trainer.train(2)
        else:
            raise AssertionError(f"unknown mode: {mode}")

        rank_payload = {
            "rank": rank,
            "first_batch": first_batch,
            "data_state": asdict(trainer.data_iterator.state()),
            "parameters": {
                name: parameter.detach().cpu()
                for name, parameter in trainer.model.named_parameters()
            },
        }
        gathered = context.all_gather_object(rank_payload)
        if context.is_main_process:
            torch.save(
                {
                    "ranks": gathered,
                    "trainer_state": asdict(trainer.trainer_state()),
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": trainer.scheduler.state_dict(),
                },
                result_path,
            )
    finally:
        context.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn(
    packed_dataset_dir: Path,
    checkpoint_dir: Path,
    result_path: Path,
    mode: str,
) -> None:
    mp.spawn(
        _worker,
        args=(
            2,
            _free_port(),
            str(packed_dataset_dir),
            str(checkpoint_dir),
            str(result_path),
            mode,
        ),
        nprocs=2,
        join=True,
    )


def _assert_nested_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert set(expected) == set(actual)
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_item, actual_item)
    else:
        assert expected == actual


def test_gloo_ddp_exact_resume_and_rank_partitioning(
    tmp_path: Path,
    packed_dataset_dir: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    continuous_path = tmp_path / "continuous.pt"
    resumed_path = tmp_path / "resumed.pt"

    _spawn(packed_dataset_dir, checkpoint_dir, continuous_path, "continuous")
    _spawn(packed_dataset_dir, checkpoint_dir, tmp_path / "unused.pt", "interrupt")
    _spawn(packed_dataset_dir, checkpoint_dir, resumed_path, "resume")

    continuous = torch.load(continuous_path, weights_only=False)
    resumed = torch.load(resumed_path, weights_only=False)
    for field, expected in continuous["trainer_state"].items():
        if field != "elapsed_training_seconds":
            assert resumed["trainer_state"][field] == expected
    _assert_nested_equal(continuous["scheduler"], resumed["scheduler"])
    _assert_nested_equal(continuous["optimizer"], resumed["optimizer"])
    assert len(list(checkpoint_dir.glob("step-*"))) == 1

    for result in (continuous, resumed):
        rank_zero, rank_one = result["ranks"]
        assert not torch.equal(rank_zero["first_batch"], rank_one["first_batch"])
        assert rank_zero["data_state"]["sample_index"] == 6
        assert rank_one["data_state"]["sample_index"] == 6
        for name, parameter in rank_zero["parameters"].items():
            assert torch.equal(parameter, rank_one["parameters"][name])

    for expected, actual in zip(
        continuous["ranks"][0]["parameters"].values(),
        resumed["ranks"][0]["parameters"].values(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
