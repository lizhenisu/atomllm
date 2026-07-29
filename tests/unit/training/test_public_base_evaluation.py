from __future__ import annotations

from atomllm.training.public_base_evaluation import quality_gate


def _report(offset: float) -> dict[str, object]:
    counts = {
        "arc_challenge": 1172,
        "ceval": 1346,
        "hellaswag": 10042,
        "mmlu": 14042,
    }
    metrics = {
        name: {
            "task_count": count,
            "chance_accuracy": 0.25,
            "raw_accuracy": 0.25 + offset,
            "normalized_accuracy": 0.25 + offset,
        }
        for name, count in counts.items()
    }
    return {
        "world_size": 6,
        "task_count": sum(counts.values()),
        "model_external_answering": False,
        "normalized_accuracy": 0.25 + offset,
        "benchmark_metrics": metrics,
    }


def test_quality_gate_requires_above_chance_full_six_gpu_result() -> None:
    gate = quality_gate(_report(0.02))

    assert gate["passed"] is True
    assert gate["checks"]["six_gpu_complete_coverage"] is True
    assert gate["checks"]["no_model_external_answering"] is True


def test_quality_gate_rejects_chance_level_model() -> None:
    gate = quality_gate(_report(0.0))

    assert gate["passed"] is False
    assert gate["checks"]["aggregate_above_chance"] is False
    assert gate["checks"]["hellaswag_minimum"] is False
