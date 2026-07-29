from atomllm.post_training.sft_probe import (
    _four_gram_repetition,
    evaluate_responses,
)


def test_response_evaluation_requires_exact_critical_answers() -> None:
    responses = {
        "capital_zh": "北京。",
        "addition_seen": "5",
        "addition_generalization": "45!",
        "multiplication": "63",
        "memory_turn_1": "好的。",
        "memory_turn_2": "你叫小明，最喜欢蓝色。",
    }

    summary = evaluate_responses(responses)

    assert summary["strict_answer_accuracy"] == 1.0
    assert summary["memory_pass"] is True
    assert summary["all_nonempty"] is True
    assert summary["repetition_gate_pass"] is True


def test_response_evaluation_rejects_verbose_or_repetitive_wrong_answer() -> None:
    repeated = "the same phrase again " * 20
    responses = {
        "capital_zh": "答案是北京",
        "addition_seen": "我认为答案是5",
        "addition_generalization": "44",
        "multiplication": "12",
        "memory_turn_1": repeated,
        "memory_turn_2": "我不记得。",
    }

    summary = evaluate_responses(responses)

    assert summary["strict_answer_accuracy"] == 0.0
    assert summary["memory_pass"] is False
    assert summary["repetition_gate_pass"] is False
    assert _four_gram_repetition(repeated) > 0.8
