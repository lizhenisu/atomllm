from atomllm.post_training.generalization_probe import evaluate_responses


def test_generalization_probe_scores_only_expected_raw_text() -> None:
    summary = evaluate_responses(
        {
            "capital_crosslingual": "渥太华",
            "fact_gold": "Au",
            "fact_author": "George Orwell",
            "instruction_exact": "OK",
            "code_unseen": "def square(n):\n    return n * n",
            "reasoning_simple": "7",
            "comparison": "0.8",
            "unseen_memory_turn_1": "好的。",
            "unseen_memory_turn_2": "你叫小华，最喜欢橙色。",
        }
    )

    assert summary["passed_count"] == summary["task_count"] == 8
    assert summary["accuracy"] == 1.0


def test_generalization_probe_rejects_incorrect_or_extra_exact_answers() -> None:
    summary = evaluate_responses(
        {
            "capital_crosslingual": "多伦多",
            "fact_gold": "答案是 Au",
            "fact_author": "Orwell",
            "instruction_exact": "OK。",
            "code_unseen": "def square(n):\n    return n + n",
            "reasoning_simple": "12",
            "comparison": "0.75",
            "unseen_memory_turn_1": "橙色",
            "unseen_memory_turn_2": "橙色",
        }
    )

    assert summary["passed_count"] == 0
    assert summary["accuracy"] == 0.0
