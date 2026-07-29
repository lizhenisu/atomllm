from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from atomllm.post_training.sft_data import (
    EOT_ID,
    IGNORE_INDEX,
    _assistant_target_windows,
    _conversation,
    _encode,
    _encode_many,
    _fragments,
    _near_duplicate_key,
    _oasst_all_assistant_paths,
)
from atomllm.post_training.sft_heldout_data import (
    build_dataset as build_heldout_dataset,
)
from atomllm.post_training.sft_heldout_data import (
    verify_dataset as verify_heldout_dataset,
)
from atomllm.post_training.sft_polish_data import (
    SFTDataError,
    _load_config as load_polish_config,
    _arithmetic,
    _basic_capabilities,
    _gsm8k_zh,
    _smoltalk,
    _stable_selected,
)
from atomllm.post_training.sft_training import (
    SFTDataset,
    SFTTrainingError,
    _block_order,
    _expected_replay_input_tokens,
    _is_replay_step,
    _packed_segment_ids,
    _verify_base_checkpoint,
    _verify_training_data_policy,
    build_training_plan,
    load_sft_config,
)
from atomllm.post_training.sft_validation import validate


def test_public_only_polish_config_rejects_synthetic_sources(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/post_training/stage8-sft-quality-v5-curated-full.yaml").read_text()
    )
    raw["data_policy"] = "public-only"
    path = tmp_path / "public-only.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SFTDataError, match="forbids locally synthesized"):
        load_polish_config(path)


def test_training_public_only_policy_rejects_unpinned_or_synthetic_data() -> None:
    public_manifest = {
        "data_policy": "public-only",
        "source_contract": {
            "smoltalk": {
                "repository": "owner/repo",
                "revision": "abc",
                "license": "Apache-2.0",
                "files": {"data.parquet": "sha"},
            }
        },
        "source_assistant_target_tokens": {"smoltalk:subset": 100},
    }
    _verify_training_data_policy(public_manifest, "public-only")

    contaminated = dict(public_manifest)
    contaminated["source_assistant_target_tokens"] = {
        "smoltalk:subset": 100,
        "synthetic-arithmetic:add": 1,
    }
    with pytest.raises(SFTTrainingError, match="synthetic target tokens"):
        _verify_training_data_policy(contaminated, "public-only")


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {"<unk>": 1, "hello": 20, "question": 21, "answer": 22},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_assistant_only_labels_exclude_prompt_and_role_tokens() -> None:
    conversation = _conversation(
        "test",
        (("user", "hello question"), ("assistant", "hello answer")),
    )
    assert conversation is not None
    tokens, labels = _encode(_tokenizer(), conversation["messages"])

    supervised = [token for token in labels if token != IGNORE_INDEX]
    assert supervised == [20, 22, EOT_ID]
    assert len(tokens) == len(labels)


def test_long_fragments_preserve_each_target_exactly_once() -> None:
    tokens = list(range(11))
    labels = [IGNORE_INDEX, *range(10)]
    fragments = list(_fragments(tokens, labels, 4))

    supervised = [
        label
        for _, fragment_labels in fragments
        for label in fragment_labels
        if label != IGNORE_INDEX
    ]
    assert supervised == list(range(10))
    assert all(len(fragment_tokens) <= 4 for fragment_tokens, _ in fragments)


def test_target_window_preserves_short_answer_and_recent_prompt_context() -> None:
    tokens = [2, *range(10, 20), 6, 90, 91, EOT_ID]
    labels = [IGNORE_INDEX] * 12 + [90, 91, EOT_ID]

    windows = _assistant_target_windows(tokens, labels, 6)

    assert windows == [
        ([2, 19, 6, 90, 91, EOT_ID], [IGNORE_INDEX] * 3 + [90, 91, EOT_ID])
    ]
    assert [
        label
        for _, window_labels in windows
        for label in window_labels
        if label != IGNORE_INDEX
    ] == [90, 91, EOT_ID]


def test_target_windows_split_oversized_answer_without_loss_or_zero_targets() -> None:
    tokens = [2, 5, 50, 6, *range(100, 112)]
    labels = [IGNORE_INDEX] * 4 + list(range(100, 112))

    windows = _assistant_target_windows(tokens, labels, 5)

    assert all(len(window_tokens) <= 5 for window_tokens, _ in windows)
    assert all(
        any(label != IGNORE_INDEX for label in window_labels)
        for _, window_labels in windows
    )
    assert [
        label
        for _, window_labels in windows
        for label in window_labels
        if label != IGNORE_INDEX
    ] == list(range(100, 112))


def test_oasst_all_paths_supervises_each_assistant_reply_once(tmp_path: Path) -> None:
    rows = [
        {
            "message_id": "p0",
            "parent_id": None,
            "message_tree_id": "tree",
            "text": "question",
            "role": "prompter",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
        {
            "message_id": "a1",
            "parent_id": "p0",
            "message_tree_id": "tree",
            "text": "answer one",
            "role": "assistant",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
        {
            "message_id": "a2",
            "parent_id": "p0",
            "message_tree_id": "tree",
            "text": "answer two",
            "role": "assistant",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
        {
            "message_id": "p1",
            "parent_id": "a1",
            "message_tree_id": "tree",
            "text": "followup",
            "role": "prompter",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
        {
            "message_id": "a3",
            "parent_id": "p1",
            "message_tree_id": "tree",
            "text": "answer three",
            "role": "assistant",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
    ]
    path = tmp_path / "oasst.parquet"
    parquet.write_table(pa.Table.from_pylist(rows), path)

    conversations = list(_oasst_all_assistant_paths(path, "oasst1"))

    assert len(conversations) == 3
    assert {
        row["messages"][-1]["content"]: row["target_message_indices"]
        for row in conversations
    } == {
        "answer one": [1],
        "answer two": [1],
        "answer three": [3],
    }
    long_branch = next(
        row for row in conversations if row["messages"][-1]["content"] == "answer three"
    )
    _, _, encoded_labels = next(_encode_many(_tokenizer(), [long_branch]))
    assert [label for label in encoded_labels if label != IGNORE_INDEX] == [
        22,
        1,
        EOT_ID,
    ]


def test_near_duplicate_key_ignores_formatting_but_preserves_roles() -> None:
    first = [{"role": "user", "content": "Hello, WORLD!"}]
    second = [{"role": "user", "content": "hello world"}]
    other_role = [{"role": "assistant", "content": "hello world"}]

    assert _near_duplicate_key(first) == _near_duplicate_key(second)
    assert _near_duplicate_key(first) != _near_duplicate_key(other_role)


def test_polish_hash_selection_is_deterministic_and_seeded() -> None:
    selected = [_stable_selected(f"row-{index}", 500, 42) for index in range(1000)]

    assert selected == [
        _stable_selected(f"row-{index}", 500, 42) for index in range(1000)
    ]
    assert selected != [
        _stable_selected(f"row-{index}", 500, 43) for index in range(1000)
    ]
    assert 30 <= sum(selected) <= 70


def test_smoltalk_can_supervise_only_the_final_assistant_turn(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source": "long-dialogue",
                    "messages": [
                        {"role": "user", "content": "question one"},
                        {"role": "assistant", "content": "answer one"},
                        {"role": "user", "content": "question two"},
                        {"role": "assistant", "content": "answer two"},
                    ],
                }
            ]
        ),
        data / "train-00000.parquet",
    )

    row = next(
        _smoltalk(
            tmp_path,
            included_sources={"long-dialogue"},
            final_assistant_only_sources={"long-dialogue"},
        )
    )

    assert row["target_message_indices"] == [3]


def test_heldout_builder_uses_test_splits_and_is_not_training_eligible(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    smol_data = raw / "smol" / "data"
    gsm_root = raw / "gsm"
    smol_data.mkdir(parents=True)
    gsm_root.mkdir()
    smol_path = smol_data / "test-000.parquet"
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source": "dialogue",
                    "messages": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ],
                }
            ]
        ),
        smol_path,
    )
    gsm_path = gsm_root / "GSM8K_zh.json"
    gsm_path.write_text(
        json.dumps(
            [
                {
                    "split": "test",
                    "question_zh": "hello question",
                    "answer_zh": "hello answer",
                }
            ]
        ),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    _tokenizer().save(str(tokenizer_path))
    config = {
        "schema_version": 1,
        "dataset_id": "heldout-test",
        "raw_root": str(raw),
        "tokenizer": str(tokenizer_path),
        "sequence_length": 16,
        "sample_seed": 7,
        "smoltalk_examples": 1,
        "final_assistant_only_sources": [],
        "sources": {
            "smoltalk": {
                "directory": "smol",
                "files": {
                    "data/test-000.parquet": hashlib.sha256(
                        smol_path.read_bytes()
                    ).hexdigest()
                },
            },
            "gsm8k_zh": {
                "directory": "gsm",
                "files": {
                    "GSM8K_zh.json": hashlib.sha256(gsm_path.read_bytes()).hexdigest()
                },
            },
        },
    }
    config_path = tmp_path / "heldout.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "heldout"

    manifest = build_heldout_dataset(config_path, output)

    assert manifest["dataset_role"] == "heldout-validation"
    assert manifest["formal_training_eligible"] is False
    assert manifest["official_test_splits_only"] is True
    assert manifest["eligible_examples"] == 2
    assert verify_heldout_dataset(output) == manifest


def test_polish_gsm_reader_keeps_test_split_out_of_training(tmp_path: Path) -> None:
    path = tmp_path / "gsm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_zh": "训练问题",
                    "answer_zh": "训练答案",
                    "split": "train",
                },
                {
                    "question_zh": "测试问题",
                    "answer_zh": "测试答案",
                    "split": "test",
                },
                {
                    "question": "fallback question",
                    "answer": "fallback answer",
                    "question_zh": "回退问题",
                    "split": "train",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = list(_gsm8k_zh(path, split="train"))

    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "训练问题"
    assert rows[1]["source"] == "gsm8k-zh:fallback-en"
    assert rows[1]["messages"][1]["content"] == "fallback answer"


def test_polish_arithmetic_is_unique_exact_and_covers_basic_addition() -> None:
    rows = list(_arithmetic())
    pairs = {
        (row["messages"][0]["content"], row["messages"][1]["content"]) for row in rows
    }

    assert len(rows) == len(pairs) == 20_872
    assert any("2 + 3" in prompt and answer == "5" for prompt, answer in pairs)


def test_polish_arithmetic_variants_increase_supervision_without_duplicates() -> None:
    rows = list(_arithmetic(variants=8))

    assert len(rows) == 20_872 * 8
    assert len({row["conversation_id"] for row in rows}) == len(rows)
    assert all("supervise_eot" not in row for row in rows)


def test_arithmetic_rows_supervise_answer_and_eot() -> None:
    row = next(_arithmetic(variants=1))
    _, _, labels = next(_encode_many(_tokenizer(), [row]))

    supervised = [label for label in labels if label != IGNORE_INDEX]
    assert supervised
    assert supervised[-1] == EOT_ID


def test_arithmetic_core_variants_cover_each_core_template() -> None:
    rows = list(_arithmetic(variants=8))[:8]
    prompts = {row["messages"][0]["content"] for row in rows}

    assert "Answer with one integer: 0 + 0 =" in prompts
    assert "直接计算 0 加 0，只输出整数。" in prompts
    assert len(prompts) == 8


def test_arithmetic_subtraction_includes_odd_results() -> None:
    match = next(
        row
        for row in _arithmetic(variants=8)
        if row["messages"][0]["content"] == "Answer with one integer: 14 - 11 ="
    )

    assert match["messages"][1]["content"] == "3"


def test_arithmetic_operations_have_balanced_core_coverage() -> None:
    counts = Counter(
        row["source"]
        for row in _arithmetic(variants=8, extended_multiply_divide=True)
    )

    assert counts == {
        "synthetic-arithmetic:add": 80_000,
        "synthetic-arithmetic:subtract": 80_088,
        "synthetic-arithmetic:multiply": 80_000,
        "synthetic-arithmetic:divide": 80_000,
    }


def test_arithmetic_can_select_multiplication_and_division_only() -> None:
    counts = Counter(
        row["source"]
        for row in _arithmetic(variants=8, operations={"multiply", "divide"})
    )

    assert counts == {
        "synthetic-arithmetic:multiply": 3_528,
        "synthetic-arithmetic:divide": 3_360,
    }


def test_near_duplicate_key_preserves_arithmetic_operators() -> None:
    add = _conversation("test", (("user", "4 + 1 = ?"), ("assistant", "5")))
    divide = _conversation("test", (("user", "4 / 1 = ?"), ("assistant", "4")))

    assert add is not None and divide is not None
    assert _near_duplicate_key(add["messages"]) != _near_duplicate_key(
        divide["messages"]
    )


def test_polish_arithmetic_extended_templates_cover_words_and_novel_forms() -> None:
    rows = [
        row
        for row in _arithmetic(variants=32)
        if row["messages"][1]["content"] == "5"
        and "2" in row["messages"][0]["content"]
        and "3" in row["messages"][0]["content"]
    ]
    prompts = {row["messages"][0]["content"] for row in rows}

    assert "直接计算 2加上3，只输出整数。" in prompts
    assert "Answer with one integer: 2 + 3 =" in prompts
    assert "What is 2 plus 3? One integer only." in prompts


def test_basic_capabilities_cover_facts_code_explanations_and_memory() -> None:
    rows = list(_basic_capabilities(variants=2))
    messages = [row["messages"] for row in rows]

    assert any(turns[-1]["content"] == "北京" for turns in messages)
    assert any("return a + b" in turns[-1]["content"] for turns in messages)
    assert any(
        "Photosynthesis uses sunlight" in turns[-1]["content"] for turns in messages
    )
    assert any(turns[-1]["content"] == "你叫小明，最喜欢蓝色。" for turns in messages)
    memory_rows = [
        row for row in rows if row["source"] == "synthetic-capability:memory"
    ]
    assert len(memory_rows) == 24 * 10 * 2
    assert all(row["target_message_indices"] == [1, 3] for row in memory_rows)


def test_training_plan_completes_full_pass_before_repeat() -> None:
    dataset = SimpleNamespace(
        block_count=4,
        target_counts=np.asarray([20, 20, 20, 20], dtype=np.int64),
    )
    config = SimpleNamespace(
        world_size=2,
        micro_batch_size=1,
        accumulation_steps=1,
        lower_target_tokens=100,
        upper_target_tokens=160,
    )

    plan = build_training_plan(dataset, config)

    assert plan.steps_per_full_pass == 2
    assert plan.total_steps == 3
    assert plan.unique_target_tokens == 80
    assert plan.repeated_target_tokens == 40
    assert plan.effective_target_tokens == 120


def test_training_plan_accounts_for_intentionally_repeated_packed_targets() -> None:
    dataset = SimpleNamespace(
        block_count=2,
        target_counts=np.asarray([40, 40], dtype=np.int64),
        unique_target_tokens=50,
    )
    config = SimpleNamespace(
        world_size=1,
        micro_batch_size=1,
        accumulation_steps=1,
        lower_target_tokens=80,
        upper_target_tokens=80,
    )

    plan = build_training_plan(dataset, config)

    assert plan.unique_target_tokens == 50
    assert plan.effective_target_tokens == 80
    assert plan.repeated_target_tokens == 30


def test_sft_block_order_is_deterministic_shuffled_and_full_coverage() -> None:
    first = _block_order(128, 20260716)
    second = _block_order(128, 20260716)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.arange(128))
    assert sorted(first.tolist()) == list(range(128))


def test_zero_target_local_batch_uses_zero_weight_supervised_surrogate() -> None:
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.inputs = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.uint32)
    dataset.labels = np.asarray([[-100, -100, -100], [-100, 21, 22]], dtype=np.int32)
    dataset.target_counts = np.asarray([0, 2], dtype=np.int64)
    dataset.fallback_target_index = 1

    inputs, labels, local_targets = dataset.batch([0], [True])

    assert local_targets == 0
    assert inputs.tolist() == [[20, 21, 22]]
    assert labels.tolist() == [[-100, 21, 22]]


def test_packed_segment_ids_reset_at_bos_and_ignore_padding() -> None:
    input_ids = torch.tensor(
        [
            [2, 5, 8, 2, 5, 8, 0, 0],
            [2, 5, 8, 0, 0, 0, 0, 0],
        ]
    )

    assert _packed_segment_ids(input_ids).tolist() == [
        [1, 1, 1, 2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
    ]


def test_training_plan_rejects_full_pass_above_upper_budget() -> None:
    dataset = SimpleNamespace(
        block_count=2,
        target_counts=np.asarray([60, 60], dtype=np.int64),
    )
    config = SimpleNamespace(
        world_size=1,
        micro_batch_size=1,
        accumulation_steps=1,
        lower_target_tokens=50,
        upper_target_tokens=100,
    )

    try:
        build_training_plan(dataset, config)
    except SFTTrainingError as error:
        assert "exceeds the configured upper budget" in str(error)
    else:
        raise AssertionError("over-budget full pass was accepted")


def test_smoke_config_accepts_a_frozen_pilot_budget(tmp_path: Path) -> None:
    config = Path(
        "configs/post_training/atom-chat-300m-sft-polish-pilot-6x3090-v1.yaml"
    )

    loaded = load_sft_config(config, tmp_path)

    assert loaded.status == "smoke"
    assert loaded.lower_target_tokens == loaded.upper_target_tokens == 17_221_716


def test_schema_v2_loads_deterministic_pretraining_replay(tmp_path: Path) -> None:
    source = Path(
        "configs/post_training/atom-chat-300m-sft-polish-pilot-6x3090-v1.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["replay"] = {
        "path": "artifacts/training-data/cooldown",
        "manifest_sha256": "a" * 64,
        "interval_steps": 4,
        "loss_weight": 0.5,
        "seed": 20260716,
    }
    raw["runtime"]["isolate_packed_conversations"] = True
    raw["runtime"]["eot_loss_weight"] = 0.25
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_sft_config(config_path, tmp_path)

    assert loaded.replay_dataset == tmp_path / "artifacts/training-data/cooldown"
    assert loaded.replay_manifest_sha256 == "a" * 64
    assert loaded.replay_loss_weight == 0.5
    assert loaded.isolate_packed_conversations is True
    assert loaded.eot_loss_weight == 0.25
    assert [_is_replay_step(loaded, step) for step in range(1, 9)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert _expected_replay_input_tokens(loaded, 4096, 3) == 0
    assert _expected_replay_input_tokens(loaded, 4096, 4) == 98_304
    assert _expected_replay_input_tokens(loaded, 4096, 9) == 196_608


def test_schema_v2_release_requires_validated_base(tmp_path: Path) -> None:
    source = Path(
        "configs/post_training/atom-chat-300m-sft-polish-pilot-6x3090-v1.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["status"] = "release"
    raw["budget"] = {
        "lower_assistant_target_tokens": 400_000_000,
        "upper_assistant_target_tokens": 800_000_000,
    }
    raw["replay"] = {
        "path": "artifacts/training-data/cooldown",
        "manifest_sha256": "a" * 64,
        "interval_steps": 4,
        "loss_weight": 0.5,
        "seed": 20260716,
    }
    config_path = tmp_path / "release.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    try:
        load_sft_config(config_path, tmp_path)
    except SFTTrainingError as error:
        assert "requires a validated base" in str(error)
    else:
        raise AssertionError("release accepted a deferred base evaluation")

    raw["initialization"]["base_validation_status"] = "passed"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_sft_config(config_path, tmp_path)
    assert loaded.base_validation_status == "passed"
    assert loaded.lower_target_tokens == 400_000_000
    assert loaded.upper_target_tokens == 800_000_000


def test_base_checkpoint_verifier_accepts_sft_protocol(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "model.safetensors": {
                        "bytes": model.stat().st_size,
                        "sha256": model_sha,
                    }
                }
            }
        )
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "COMPLETE").write_text("atomllm-sft-checkpoint-v1\n")

    _verify_base_checkpoint(
        tmp_path,
        expected_manifest_sha256=manifest_sha,
        expected_model_sha256=model_sha,
    )


def _validation_run(
    root: Path,
    *,
    steps: int,
    restored_from_step: int | None,
    initial_loss: float,
    final_loss: float,
    base_model_sha256: str | None,
) -> Path:
    checkpoint = root / "checkpoints" / f"step-{steps:09d}"
    reports = root / "reports"
    checkpoint.mkdir(parents=True)
    reports.mkdir()
    payload = checkpoint / "payload.bin"
    payload.write_bytes(b"synthetic-checkpoint")
    payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "files": {
            "payload.bin": {"bytes": payload.stat().st_size, "sha256": payload_sha}
        }
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "COMPLETE").write_text("complete\n")
    (root / "checkpoints" / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint.name,
                "manifest_sha256": manifest_sha,
            }
        )
    )
    report = {
        "completed_steps": steps,
        "restored_from_step": restored_from_step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "formal_completion_reached": False,
        "release_plan": {"base_model_sha256": base_model_sha256},
    }
    (reports / "training-report.json").write_text(json.dumps(report))
    if restored_from_step is not None:
        (reports / "resume-events.jsonl").write_text(
            json.dumps(
                {
                    "restored_from_step": restored_from_step,
                    "target_step": steps,
                }
            )
            + "\n"
        )
    return root


def test_stage8b_validator_freezes_completion_evidence(tmp_path: Path) -> None:
    overfit = _validation_run(
        tmp_path / "overfit",
        steps=20,
        restored_from_step=None,
        initial_loss=10.0,
        final_loss=5.0,
        base_model_sha256=None,
    )
    ddp = _validation_run(
        tmp_path / "ddp",
        steps=30,
        restored_from_step=10,
        initial_loss=2.5,
        final_loss=2.2,
        base_model_sha256=(
            "6017a1d5a3e95a13be9c9ad38f5bc51b9528981ea10b555f873779fdfdb662c7"
        ),
    )
    output = tmp_path / "validation"

    report = validate(overfit, ddp, output)

    assert report["status"] == "passed"
    assert (output / "COMPLETED").is_file()
