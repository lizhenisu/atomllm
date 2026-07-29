import re

import pytest

from atomllm.post_training.arithmetic_probe import (
    _heldout_tasks,
    _normalized_integer,
    _tasks,
)


def test_arithmetic_probe_is_balanced_and_deterministic() -> None:
    first = _tasks()
    second = _tasks()

    assert first == second
    assert len(first) == 100
    assert {
        name: sum(task[0] == name for task in first) for name in {t[0] for t in first}
    } == {
        "add": 25,
        "subtract": 25,
        "multiply": 25,
        "divide": 25,
    }


def test_arithmetic_probe_requires_one_exact_integer() -> None:
    assert _normalized_integer(" 45。") == "45"
    assert _normalized_integer("-3") == "-3"
    assert _normalized_integer("045") == "45"
    assert _normalized_integer("答案是45") is None
    assert _normalized_integer("45 46") is None


def test_heldout_arithmetic_is_balanced_by_split_and_operation() -> None:
    tasks = _heldout_tasks()

    assert len(tasks) == 200
    assert len({(split, prompt) for split, _, prompt, _ in tasks}) == len(tasks)
    assert {
        (split, operation): sum(
            task_split == split and task_operation == operation
            for task_split, task_operation, _, _ in tasks
        )
        for split in {task[0] for task in tasks}
        for operation in {task[1] for task in tasks}
    } == {
        (split, operation): 25
        for split in ("template_holdout", "range_holdout")
        for operation in ("add", "subtract", "multiply", "divide")
    }


def test_range_holdout_operands_are_outside_training_ranges() -> None:
    tasks = [task for task in _heldout_tasks() if task[0] == "range_holdout"]

    assert len(tasks) == 100
    for _, operation, prompt, _ in tasks:
        operands = [int(value) for value in re.findall(r"\d+", prompt)]
        minimum = 101 if operation in {"add", "subtract"} else 21
        assert operands
        assert all(value >= minimum for value in operands)


def test_arithmetic_probe_rejects_unknown_suite() -> None:
    from atomllm.post_training.arithmetic_probe import run

    with pytest.raises(ValueError, match="unsupported arithmetic probe suite"):
        run(None, None, None, suite="unknown")  # type: ignore[arg-type]
