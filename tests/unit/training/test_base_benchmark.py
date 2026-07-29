from atomllm.training.base_benchmark import _stable_sample, _task


def test_stable_sample_is_deterministic_and_seeded() -> None:
    rows = [{"id": str(index)} for index in range(100)]

    first = _stable_sample(rows, 10, seed=42, key="id")
    second = _stable_sample(rows, 10, seed=42, key="id")
    different = _stable_sample(rows, 10, seed=43, key="id")

    assert first == second
    assert first != different
    assert len(first) == 10


def test_public_task_requires_a_valid_answer() -> None:
    task = _task("suite", "id", "prompt", ["a", "b"], 1)

    assert task["answer"] == 1
    assert task["task_id"] == "suite:id"
