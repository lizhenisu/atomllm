import hashlib
import json
from pathlib import Path

from atomllm.training.public_smoke_gate import verify_smoke


def test_smoke_gate_requires_complete_six_gpu_real_data_run(
    tmp_path, monkeypatch
) -> None:
    config_source = Path(
        "configs/training/atom-base-300m-recovery-public-4k-6x3090-v1.yaml"
    )
    config_path = tmp_path / "training.yaml"
    config_path.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/model").mkdir(parents=True)
    (tmp_path / "configs/model/atom-base-300m-long-v1.yaml").write_text(
        Path("configs/model/atom-base-300m-long-v1.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "configs/training").mkdir(parents=True)
    (
        tmp_path / "configs/training/atom-base-300m-cooldown-4k-6x3090-v1.yaml"
    ).write_text(
        Path("configs/training/atom-base-300m-cooldown-4k-6x3090-v1.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    checkpoint_dir = run / "checkpoints/step-000000030"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_manifest = checkpoint_dir / "manifest.json"
    checkpoint_manifest.write_text("{}\n", encoding="utf-8")
    expected_tokens = 30 * 196_608
    report = {
        "trainer_state": {"global_step": 30, "tokens_seen": expected_tokens},
        "step_metrics": [
            {
                "global_step": index + 1,
                "loss": 10.0 - index * 0.01,
                "gradient_norm": 1.0,
            }
            for index in range(30)
        ],
        "distributed": {"world_size": 6},
        "tokens_per_second": 70000.0,
        "peak_reserved_gib": 18.0,
    }
    training_report = run / "reports/training-report-test.json"
    training_report.write_text(json.dumps(report) + "\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "atomllm.training.public_smoke_gate.verify_checkpoint_directory",
        lambda path: {
            "global_step": 20 if path.name.endswith("020") else 30,
            "tokens_seen": 20 * 196_608
            if path.name.endswith("020")
            else expected_tokens,
            "config_sha256": config_sha,
        },
    )

    gate = verify_smoke(
        run_dir=Path("run"),
        training_config=Path("training.yaml"),
        output_dir=Path("gate"),
        project_root=tmp_path,
    )

    assert gate["full_training_eligible"] is True
    assert gate["tokens_seen"] == expected_tokens
    assert gate["checks"]["model_external_capability"] is False


def test_smoke_gate_accepts_exact_resume_metric_suffix(tmp_path, monkeypatch) -> None:
    config_source = Path(
        "configs/training/atom-base-300m-recovery-public-4k-6x3090-v1.yaml"
    )
    config_path = tmp_path / "training.yaml"
    config_path.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/model").mkdir(parents=True)
    (tmp_path / "configs/model/atom-base-300m-long-v1.yaml").write_text(
        Path("configs/model/atom-base-300m-long-v1.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "configs/training").mkdir(parents=True)
    (
        tmp_path / "configs/training/atom-base-300m-cooldown-4k-6x3090-v1.yaml"
    ).write_text(
        Path("configs/training/atom-base-300m-cooldown-4k-6x3090-v1.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    checkpoint_dir = run / "checkpoints/step-000000030"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    expected_tokens = 30 * 196_608
    report = {
        "trainer_state": {"global_step": 30, "tokens_seen": expected_tokens},
        "restored_checkpoint_id": "step-000000020",
        "restored_global_step": 20,
        "step_metrics": [
            {
                "global_step": step,
                "loss": 9.0 - (step - 21) * 0.01,
                "gradient_norm": 1.0,
            }
            for step in range(21, 31)
        ],
        "distributed": {"world_size": 6},
        "tokens_per_second": 70000.0,
        "peak_reserved_gib": 18.0,
    }
    training_report = run / "reports/training-report-resumed.json"
    training_report.write_text(json.dumps(report) + "\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "atomllm.training.public_smoke_gate.verify_checkpoint_directory",
        lambda path: {
            "global_step": 20 if path.name.endswith("020") else 30,
            "tokens_seen": 20 * 196_608
            if path.name.endswith("020")
            else expected_tokens,
            "config_sha256": config_sha,
        },
    )

    gate = verify_smoke(
        run_dir=Path("run"),
        training_config=Path("training.yaml"),
        output_dir=Path("gate"),
        project_root=tmp_path,
    )

    assert gate["full_training_eligible"] is True
    assert gate["restored_global_step"] == 20
    assert gate["checks"]["continuous_process_metric_suffix"] is True
