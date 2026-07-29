import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomllm.training.config import load_training_config
from atomllm.training.public_pretraining_release import build_release


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("vocab_size", "parameter_count"),
    [(32000, 303_350_784), (48000, 319_734_784)],
)
def test_release_freezes_selected_tokenizer_and_full_data_budget(
    tmp_path, monkeypatch, vocab_size: int, parameter_count: int
) -> None:
    (tmp_path / "base-model.yaml").write_text(
        Path("configs/model/atom-base-300m.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_file.write_text("{}\n", encoding="utf-8")
    tokenizer_manifest = tokenizer_dir / "manifest.json"
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection = {
        "training_eligible": True,
        "gpu_confirmed": True,
        "selected_tokenizer_dir": "tokenizer",
        "selected_tokenizer_sha256": _sha256(tokenizer_file),
        "selected_tokenizer_manifest_sha256": _sha256(tokenizer_manifest),
        "selected_parameter_count": parameter_count,
    }
    selection_report = selection_dir / "report.json"
    selection_report.write_text(json.dumps(selection) + "\n", encoding="utf-8")
    selection_sha = _sha256(selection_report)
    (selection_dir / "COMPLETED").write_text(
        f"{selection_sha}  report.json\n", encoding="utf-8"
    )
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    (shards_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    token_count = 508_627 * 48 * 4096
    shards = {
        "dataset_id": "public-token-shards-test",
        "identity_sha256": "f" * 64,
        "identity": {
            "plan_sha256": "a" * 64,
            "tokenizer_sha256": _sha256(tokenizer_file),
            "tokenizer_manifest_sha256": _sha256(tokenizer_manifest),
            "gpu_selection_report_sha256": selection_sha,
            "validation_exclusion": None,
            "training_split": "all-selected-documents",
            "validation_status": "deferred",
        },
        "tokenizer": {
            "vocab_size": vocab_size,
            "tokenizer_sha256": _sha256(tokenizer_file),
        },
        "sequence_length": 4096,
        "content_token_count": 100_000_000_000,
        "token_count": token_count,
        "language_content_tokens": {
            "en": 50_000_000_000,
            "code": 10_000_000_000,
            "zh-Hans": 40_000_000_000,
        },
    }
    monkeypatch.setattr(
        "atomllm.training.public_pretraining_release.verify_formal_token_shards",
        lambda _path: shards,
    )
    monkeypatch.setattr(
        "atomllm.training.public_pretraining_release.verify_tokenizer_directory",
        lambda _path: (
            SimpleNamespace(get_vocab_size=lambda **_kwargs: vocab_size),
            {"vocab_size": vocab_size},
            tokenizer_manifest,
        ),
    )

    release = build_release(
        tokenizer_selection_dir=Path("selection"),
        token_shards_dir=Path("shards"),
        base_model_config=Path("base-model.yaml"),
        output_dir=Path("release"),
        project_root=tmp_path,
    )

    assert release["training_eligible"] is True
    assert release["training_config"]["total_steps"] == 508_627
    assert release["training_config"]["expected_total_tokens"] == token_count
    assert release["checks"]["synthetic_training_content"] is False
    assert release["checks"]["validation_deferred"] is True
    assert release["validation"] == {"status": "deferred", "dataset": None}
    assert release["model_config"]["vocab_size"] == vocab_size
    assert release["model_config"]["parameter_count"] == parameter_count
    config = load_training_config(
        tmp_path / "release/training.yaml", project_root=tmp_path
    )
    assert config.scheduler.name == "trapezoidal"
    assert config.runtime.ddp_bucket_cap_mb == 50
    assert config.runtime.deterministic is False
    assert release["training_config"]["gpu_deterministic"] is False
    assert config.budget is not None
    assert config.budget.minimum_coverage_ratio == 0.99999
