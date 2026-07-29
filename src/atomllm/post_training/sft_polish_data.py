"""Build a deterministic, independently auditable stage-8 quality Pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow.parquet as parquet
import yaml
from tokenizers import Tokenizer

from atomllm.post_training.sft_data import (
    IGNORE_INDEX,
    INPUT_FILE,
    LABEL_FILE,
    PAD_ID,
    SCHEMA_VERSION,
    SFTDataError,
    _assistant_target_windows,
    _coig,
    _conversation,
    _encode_many,
    _near_duplicate_key,
    _sha256,
    verify_dataset,
)


def _stable_selected(conversation_id: str, basis_points: int, seed: int) -> bool:
    """Select independently of source order and shard layout."""
    digest = hashlib.sha256(f"{seed}:{conversation_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10_000 < basis_points


def _smoltalk(
    root: Path,
    *,
    split: str = "train",
    included_sources: set[str] | None = None,
    final_assistant_only_sources: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    if split not in {"train", "test"}:
        raise SFTDataError("SmolTalk split must be train or test")
    final_only = final_assistant_only_sources or set()
    for path in sorted((root / "data").glob(f"{split}-*.parquet")):
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(columns=["messages", "source"], batch_size=2048):
            for item in batch.to_pylist():
                if (
                    included_sources is not None
                    and item["source"] not in included_sources
                ):
                    continue
                row = _conversation(
                    f"smoltalk:{item['source']}",
                    (
                        (message["role"], message["content"])
                        for message in item["messages"]
                    ),
                    final_assistant_only=item["source"] in final_only,
                )
                if row is not None:
                    yield row


def _metamath(path: Path) -> Iterator[dict[str, Any]]:
    for item in json.loads(path.read_text(encoding="utf-8")):
        row = _conversation(
            f"metamath:{item['type']}",
            (("user", item["query"]), ("assistant", item["response"])),
        )
        if row is not None:
            yield row


def _gsm8k_zh(path: Path, *, split: str) -> Iterator[dict[str, Any]]:
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item.get("split") != split:
            continue
        question = item.get("question_zh") or item.get("question")
        answer = item.get("answer_zh") or item.get("answer")
        fallback = not item.get("question_zh") or not item.get("answer_zh")
        row = _conversation(
            "gsm8k-zh:fallback-en" if fallback else "gsm8k-zh",
            (("user", question), ("assistant", answer)),
        )
        if row is not None:
            yield row


def _arithmetic(
    *,
    variants: int = 1,
    operations: set[str] | None = None,
    extended_multiply_divide: bool = False,
) -> Iterator[dict[str, Any]]:
    """Generate exact short-answer arithmetic without external model output."""
    allowed_operations = {"add", "subtract", "multiply", "divide"}
    selected_operations = allowed_operations if operations is None else operations
    if not selected_operations or not selected_operations <= allowed_operations:
        raise SFTDataError("synthetic arithmetic operations are invalid")
    multiply_limit = 100 if extended_multiply_divide else 21
    divisor_limit = 101 if extended_multiply_divide else 21
    quotient_limit = 100 if extended_multiply_divide else 21
    examples: Iterable[tuple[str, int, int, int]] = (
        [
            ("add", left, right, left + right)
            for left in range(100)
            for right in range(100)
        ]
        + [
            ("subtract", left, right, left - right)
            for left in range(141)
            for right in range(left + 1)
        ]
        + [
            ("multiply", left, right, left * right)
            for left in range(multiply_limit)
            for right in range(multiply_limit)
        ]
        + [
            ("divide", divisor * quotient, divisor, quotient)
            for divisor in range(1, divisor_limit)
            for quotient in range(quotient_limit)
        ]
    )
    symbols = {"add": "+", "subtract": "-", "multiply": "×", "divide": "÷"}
    ascii_symbols = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
    }
    operation_en = {
        "add": "plus",
        "subtract": "minus",
        "multiply": "times",
        "divide": "divided by",
    }
    operation_zh = {
        "add": "加上",
        "subtract": "减去",
        "multiply": "乘以",
        "divide": "除以",
    }
    operation_zh_short = {
        "add": "加",
        "subtract": "减",
        "multiply": "乘",
        "divide": "除",
    }
    if not 1 <= variants <= 32:
        raise SFTDataError("synthetic arithmetic variants must be in [1, 32]")
    templates = (
        "Calculate {left} {ascii_symbol} {right}. Give only the answer.",
        "Answer with one integer: {left} {ascii_symbol} {right} =",
        "计算 {left} {symbol} {right}。只给出答案。",
        "直接计算 {left} {operation_zh_short} {right}，只输出整数。",
        "What is {left} {operation_en} {right}? One integer only.",
        "{left}{operation_zh}{right}是多少？答案只写整数。",
        "Compute {left} {symbol} {right}. Answer concisely.",
        "请计算：{left} {symbol} {right}。简洁作答。",
        "What is {left} {symbol} {right}? Reply with only the number.",
        "Solve: {left} {symbol} {right}. Output the result only.",
        "{left} {symbol} {right} 等于多少？只回答数字。",
        "算出 {left} {symbol} {right} 的结果，只输出结果。",
        "Return the exact integer for {left} {symbol} {right}.",
        "Evaluate {left} {symbol} {right}; output only the integer.",
        "Give only the integer value of {left} {symbol} {right}.",
        "Please solve {left} {symbol} {right} and reply with one number.",
        "Compute the expression {left} {symbol} {right}. Integer only.",
        "Respond with the numeric result: {left} {symbol} {right}.",
        "Calculate {left} {operation_en} {right}; return the number alone.",
        "The result of {left} {operation_en} {right} is what integer?",
        "Solve {left} {operation_en} {right}. Do not include an explanation.",
        "Work out {left} {operation_en} {right}; answer with digits only.",
        "Provide the integer obtained from {left} {operation_en} {right}.",
        "Exactly one number: what is {left} {operation_en} {right}?",
        "Quick arithmetic: {left} {operation_en} {right}. Number only.",
        "直接计算 {left}{operation_zh}{right}，只输出整数。",
        "求 {left} {symbol} {right} 的值，仅回复数字。",
        "请给出算式 {left} {symbol} {right} 的整数结果。",
        "口算 {left}{operation_zh}{right}，不要解释。",
        "准确回答：{left}{operation_zh}{right}，只写一个数。",
        "以下运算的结果是什么：{left} {symbol} {right}？仅写整数。",
        "完成计算 {left}{operation_zh}{right}。答案只包含数字。",
    )
    for operation, left, right, answer in examples:
        if operation not in selected_operations:
            continue
        symbol = symbols[operation]
        ascii_symbol = ascii_symbols[operation]
        key = f"{operation}:{left}:{right}"
        template_count = 8 if variants <= 8 else len(templates)
        offset = int(hashlib.sha256(key.encode()).hexdigest()[:2], 16) % template_count
        for variant in range(variants):
            template = templates[(offset + variant) % template_count]
            prompt = template.format(
                left=left,
                symbol=symbol,
                ascii_symbol=ascii_symbol,
                right=right,
                operation_en=operation_en[operation],
                operation_zh=operation_zh[operation],
                operation_zh_short=operation_zh_short[operation],
            )
            row = _conversation(
                f"synthetic-arithmetic:{operation}",
                (("user", prompt), ("assistant", str(answer))),
            )
            if row is not None:
                yield row


def _basic_capabilities(*, variants: int = 16) -> Iterator[dict[str, Any]]:
    """Generate concise factual, explanatory, coding, and memory supervision."""
    if not 1 <= variants <= 16:
        raise SFTDataError("synthetic capability variants must be in [1, 16]")
    wrappers = (
        "{question}",
        "请准确回答：{question}",
        "请简洁作答：{question}",
        "直接给出答案：{question}",
        "回答下面的问题，不要添加无关内容：{question}",
        "任务：{question}",
        "问题：{question}\n答案：",
        "请根据常识回答：{question}",
        "Answer accurately: {question}",
        "Answer concisely: {question}",
        "Give the direct answer: {question}",
        "Respond without unrelated details: {question}",
        "Task: {question}",
        "Question: {question}\nAnswer:",
        "Provide a correct and concise response: {question}",
        "Complete this request carefully: {question}",
    )
    facts = (
        ("中国的首都是哪里？", "北京"),
        ("法国的首都是哪里？", "巴黎"),
        ("日本的首都是哪里？", "东京"),
        ("德国的首都是哪里？", "柏林"),
        ("意大利的首都是哪里？", "罗马"),
        ("英国的首都是哪里？", "伦敦"),
        ("地球的天然卫星是什么？", "月球"),
        ("太阳系最大的行星是什么？", "木星"),
        ("被称为红色星球的是哪颗行星？", "火星"),
        ("水的化学式是什么？", "H2O"),
        ("标准大气压下水的沸点是多少摄氏度？", "100摄氏度"),
        ("一千米等于多少米？", "1000米"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is Earth's natural satellite?", "The Moon"),
        ("Which planet is the largest in the Solar System?", "Jupiter"),
        ("What gas do plants absorb for photosynthesis?", "Carbon dioxide"),
        ("How many meters are in one kilometer?", "1000 meters"),
    )
    explanations = (
        (
            "用两句话向初学者解释什么是机器学习。",
            "机器学习让计算机从数据中发现规律，而不是为每种情况手写规则。训练好的模型可以利用这些规律对新数据进行预测或决策。",
        ),
        (
            "Explain photosynthesis in two concise sentences.",
            "Photosynthesis uses sunlight to convert carbon dioxide and water into sugars. Plants store this chemical energy and release oxygen as a by-product.",
        ),
        (
            "Name three renewable energy sources.",
            "Solar energy, wind energy, and hydroelectric power.",
        ),
        (
            "用两句话解释互联网是什么。",
            "互联网是连接全球计算机网络的基础设施。它让设备按照通用协议交换数据并提供网页、通信等服务。",
        ),
    )
    code_tasks = (
        (
            "Write a Python function add(a, b) that returns their sum.",
            "```python\ndef add(a, b):\n    return a + b\n```",
        ),
        (
            "Write a Python function subtract(a, b) that returns a minus b.",
            "```python\ndef subtract(a, b):\n    return a - b\n```",
        ),
        (
            "Write a Python function is_even(n) that returns whether n is even.",
            "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        ),
        (
            "Write a Python function reverse_text(text) that reverses a string.",
            "```python\ndef reverse_text(text):\n    return text[::-1]\n```",
        ),
    )
    for category, records in (
        ("fact", facts),
        ("explanation", explanations),
        ("code", code_tasks),
    ):
        for question, answer in records:
            for variant, wrapper in enumerate(wrappers[:variants]):
                prompt = wrapper.format(question=question)
                row = _conversation(
                    f"synthetic-capability:{category}",
                    (("user", prompt), ("assistant", answer)),
                )
                if row is not None:
                    yield row

    names = (
        "小明",
        "小红",
        "小林",
        "小周",
        "小陈",
        "小王",
        "小李",
        "小张",
        "小刘",
        "小杨",
        "子涵",
        "子轩",
        "欣怡",
        "雨桐",
        "晨曦",
        "嘉怡",
        "Alex",
        "Taylor",
        "Jordan",
        "Morgan",
        "Casey",
        "Riley",
        "Jamie",
        "Robin",
    )
    colors = (
        "蓝色",
        "红色",
        "绿色",
        "黄色",
        "紫色",
        "橙色",
        "白色",
        "黑色",
        "粉色",
        "灰色",
    )
    memory_templates = (
        (
            "记住：我叫{name}，我最喜欢{color}。",
            "好的，我记住了。",
            "我叫什么，最喜欢什么颜色？",
            "你叫{name}，最喜欢{color}。",
        ),
        (
            "请记住，我的名字是{name}，最喜欢的颜色是{color}。",
            "明白，我会记住。",
            "请告诉我的名字和喜欢的颜色。",
            "你的名字是{name}，喜欢{color}。",
        ),
        (
            "在这次对话中记下：{name}喜欢{color}。",
            "已记下。",
            "刚才的名字和颜色分别是什么？",
            "名字是{name}，颜色是{color}。",
        ),
        (
            "我叫{name}，偏爱{color}。稍后我会问你。",
            "好的。",
            "你还记得我的名字和偏爱的颜色吗？",
            "记得，你叫{name}，偏爱{color}。",
        ),
        (
            "Remember that my name is {name} and my favorite color is {color}.",
            "Understood. I will remember that.",
            "What is my name and favorite color?",
            "Your name is {name}, and your favorite color is {color}.",
        ),
        (
            "For this conversation, associate {name} with {color}.",
            "Got it.",
            "Which name and color did I ask you to remember?",
            "You asked me to remember {name} and {color}.",
        ),
    )
    memory_variant_count = min(variants, len(memory_templates))
    for name in names:
        for color in colors:
            for template in memory_templates[:memory_variant_count]:
                first_user, first_assistant, second_user, second_assistant = (
                    text.format(name=name, color=color) for text in template
                )
                row = _conversation(
                    "synthetic-capability:memory",
                    (
                        ("user", first_user),
                        ("assistant", first_assistant),
                        ("user", second_user),
                        ("assistant", second_assistant),
                    ),
                )
                if row is not None:
                    yield row


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "dataset_id",
        "raw_root",
        "tokenizer",
        "sequence_length",
        "selection_seed",
        "sources",
    }
    optional = {"data_policy"}
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or not set(config) <= required | optional
    ):
        raise SFTDataError(
            f"polish config fields must be {sorted(required)} with optional "
            f"{sorted(optional)}"
        )
    if config["schema_version"] not in {1, 2}:
        raise SFTDataError("unsupported polish data schema")
    if not isinstance(config["sequence_length"], int) or config["sequence_length"] < 2:
        raise SFTDataError("sequence_length must be at least 2")
    data_policy = config.get("data_policy", "mixed")
    if data_policy not in {"mixed", "public-only"}:
        raise SFTDataError("unsupported SFT data policy")
    if data_policy == "public-only" and any(
        name.startswith("synthetic_") for name in config["sources"]
    ):
        raise SFTDataError("public-only SFT forbids locally synthesized sources")
    required_sources = ["smoltalk", "metamath", "gsm8k_zh"]
    if config["schema_version"] == 2:
        required_sources.append("coig")
    for name in required_sources:
        source = config["sources"].get(name)
        if not isinstance(source, dict):
            raise SFTDataError(f"missing source config: {name}")
        if not 0 <= source["selection_basis_points"] <= 10_000:
            raise SFTDataError(f"invalid selection_basis_points for {name}")
        validation_basis_points = source.get("validation_basis_points", 0)
        if (
            type(validation_basis_points) is not int
            or not 0 <= validation_basis_points < 10_000
        ):
            raise SFTDataError(f"invalid validation_basis_points for {name}")
    smoltalk = config["sources"]["smoltalk"]
    per_source = smoltalk.get("source_selection_basis_points", {})
    if not isinstance(per_source, dict) or any(
        type(value) is not int or not 0 < value <= 10_000
        for value in per_source.values()
    ):
        raise SFTDataError("invalid SmolTalk per-source selection basis points")
    included = set(smoltalk.get("included_sources", []))
    final_only = set(smoltalk.get("final_assistant_only_sources", []))
    if not final_only <= included:
        raise SFTDataError("final-only SmolTalk sources must be included")
    if not set(per_source) <= included:
        raise SFTDataError("SmolTalk per-source quotas must be included")
    return config


def _verify_inputs(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for source_name, source in sources.items():
        if source_name in {"synthetic_arithmetic", "synthetic_capabilities"}:
            verified[source_name] = source
            continue
        source_root = root / source["directory"]
        files = source["files"]
        for relative, expected_sha in files.items():
            path = source_root / relative
            if not path.is_file() or _sha256(path) != expected_sha:
                raise SFTDataError(f"source file hash mismatch: {path}")
        verified[source_name] = {
            "repository": source["repository"],
            "revision": source["revision"],
            "license": source["license"],
            "selection_basis_points": source["selection_basis_points"],
            "files": files,
        }
        if "included_sources" in source:
            verified[source_name]["included_sources"] = source["included_sources"]
        for field in (
            "source_selection_basis_points",
            "final_assistant_only_sources",
            "validation_basis_points",
        ):
            if field in source:
                verified[source_name][field] = source[field]
    return verified


def _deduplicate(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    exact: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: (item["conversation_id"], item["source"])):
        if row["conversation_id"] in exact:
            rejected[f"{row['source']}:exact_duplicate"] += 1
        else:
            exact[row["conversation_id"]] = row
    near: dict[str, dict[str, Any]] = {}
    for row in exact.values():
        key = _near_duplicate_key(row["messages"])
        if key in near:
            rejected[f"{row['source']}:near_duplicate"] += 1
        else:
            near[key] = row
    return sorted(near.values(), key=lambda item: item["conversation_id"]), rejected


def build_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    root = Path(config["raw_root"])
    tokenizer_path = Path(config["tokenizer"])
    if not root.is_dir() or not tokenizer_path.is_file():
        raise SFTDataError("polish raw root or tokenizer is missing")
    verified_sources = _verify_inputs(root, config["sources"])
    seed = int(config["selection_seed"])

    source_config = config["sources"]
    candidates: list[dict[str, Any]] = []
    normalized: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    reserved_validation: Counter[str] = Counter()

    readers = {
        "smoltalk": _smoltalk(
            root / source_config["smoltalk"]["directory"],
            included_sources=(
                set(source_config["smoltalk"]["included_sources"])
                if "included_sources" in source_config["smoltalk"]
                else None
            ),
            final_assistant_only_sources=set(
                source_config["smoltalk"].get("final_assistant_only_sources", [])
            ),
        ),
        "metamath": _metamath(
            root / source_config["metamath"]["directory"] / "MetaMathQA-40K.json"
        ),
        "gsm8k_zh": _gsm8k_zh(
            root / source_config["gsm8k_zh"]["directory"] / "GSM8K_zh.json",
            split="train",
        ),
    }
    if "coig" in source_config:
        readers["coig"] = _coig(root / source_config["coig"]["directory"])
    for source_name, rows in readers.items():
        for row in rows:
            normalized[source_name] += 1
            validation_basis_points = source_config[source_name].get(
                "validation_basis_points", 0
            )
            if validation_basis_points and _stable_selected(
                row["conversation_id"], validation_basis_points, seed + 1
            ):
                reserved_validation[source_name] += 1
                continue
            basis_points = source_config[source_name]["selection_basis_points"]
            if source_name == "smoltalk":
                subset = row["source"].removeprefix("smoltalk:")
                basis_points = (
                    source_config[source_name]
                    .get("source_selection_basis_points", {})
                    .get(subset, basis_points)
                )
            if _stable_selected(row["conversation_id"], basis_points, seed):
                selected[source_name] += 1
                candidates.append(row)
    if "synthetic_arithmetic" in source_config:
        arithmetic_config = source_config["synthetic_arithmetic"]
        arithmetic_rows = list(
            _arithmetic(
                variants=int(arithmetic_config.get("variants", 1)),
                operations=set(arithmetic_config.get("operations", [])) or None,
                extended_multiply_divide=bool(
                    arithmetic_config.get("extended_multiply_divide", False)
                ),
            )
        )
        normalized["synthetic_arithmetic"] = len(arithmetic_rows)
        selected["synthetic_arithmetic"] = len(arithmetic_rows)
        candidates.extend(arithmetic_rows)
    if "synthetic_capabilities" in source_config:
        capability_rows = list(
            _basic_capabilities(
                variants=int(
                    source_config["synthetic_capabilities"].get("variants", 16)
                )
            )
        )
        normalized["synthetic_capabilities"] = len(capability_rows)
        selected["synthetic_capabilities"] = len(capability_rows)
        candidates.extend(capability_rows)
    train, rejected = _deduplicate(candidates)
    if not train:
        raise SFTDataError("no eligible polish conversations")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequence_length = config["sequence_length"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    source_examples: Counter[str] = Counter()
    source_targets: Counter[str] = Counter()
    current_tokens: list[int] = []
    current_labels: list[int] = []
    block_count = 0
    unique_targets = 0
    long_conversations = 0
    max_conversation_tokens = 0

    def flush(inputs: Any, labels_out: Any) -> None:
        nonlocal block_count, current_tokens, current_labels
        if not current_tokens:
            return
        padding = sequence_length - len(current_tokens)
        np.asarray(current_tokens + [PAD_ID] * padding, dtype="<u4").tofile(inputs)
        np.asarray(current_labels + [IGNORE_INDEX] * padding, dtype="<i4").tofile(
            labels_out
        )
        block_count += 1
        current_tokens = []
        current_labels = []

    try:
        with (
            (temporary / INPUT_FILE).open("wb") as inputs,
            (temporary / LABEL_FILE).open("wb") as labels_out,
        ):
            for start in range(0, len(train), 512):
                for row, tokens, labels in _encode_many(
                    tokenizer, train[start : start + 512]
                ):
                    target_count = sum(label != IGNORE_INDEX for label in labels)
                    if not target_count:
                        rejected[f"{row['source']}:no_assistant_targets"] += 1
                        continue
                    source_examples[row["source"]] += 1
                    source_targets[row["source"]] += target_count
                    unique_targets += target_count
                    max_conversation_tokens = max(max_conversation_tokens, len(tokens))
                    long_conversations += len(tokens) > sequence_length
                    for window_tokens, window_labels in _assistant_target_windows(
                        tokens, labels, sequence_length
                    ):
                        if len(current_tokens) + len(window_tokens) > sequence_length:
                            flush(inputs, labels_out)
                        current_tokens.extend(window_tokens)
                        current_labels.extend(window_labels)
            flush(inputs, labels_out)
        files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256(temporary / name),
            }
            for name in (INPUT_FILE, LABEL_FILE)
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": config["dataset_id"],
            "data_policy": config.get("data_policy", "mixed"),
            "created_at": datetime.now(UTC).isoformat(),
            "config_sha256": _sha256(config_path),
            "tokenizer_sha256": _sha256(tokenizer_path),
            "selection_seed": seed,
            "selection_method": "sha256(seed:conversation_id)-basis-points",
            "source_contract": verified_sources,
            "sequence_length": sequence_length,
            "block_count": block_count,
            "eligible_examples": sum(source_examples.values()),
            "unique_assistant_target_tokens": unique_targets,
            "validation_examples": 1319,
            "source_normalized_candidates": dict(sorted(normalized.items())),
            "source_selected_candidates": dict(sorted(selected.items())),
            "source_reserved_validation_examples": dict(
                sorted(reserved_validation.items())
            ),
            "source_eligible_examples": dict(sorted(source_examples.items())),
            "source_assistant_target_tokens": dict(sorted(source_targets.items())),
            "source_validation_examples": {"gsm8k-zh": 1319},
            "rejected_examples": dict(sorted(rejected.items())),
            "long_conversations_sliced": long_conversations,
            "max_conversation_tokens": max_conversation_tokens,
            "silent_truncation_count": 0,
            "windowing_strategy": "assistant_target_windows",
            "packing": "stable-sha-order-no-target-truncation",
            "assistant_only_loss": True,
            "builder_sha256": _sha256(Path(__file__)),
            "formal_training_eligible": True,
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_sha = _sha256(temporary / "manifest.json")
        (temporary / "COMPLETED").write_text(
            f"manifest_sha256={manifest_sha}\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise SFTDataError(f"output already exists: {output_dir}")
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = (
        verify_dataset(args.output_dir)
        if args.verify_only
        else build_dataset(args.config, args.output_dir)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
