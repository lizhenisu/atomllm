import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import atomllm.tokenizer.audit as audit
from atomllm.tokenizer.audit import TokenizerAuditError, build_tokenizer_smoke_version
from atomllm.tokenizer.config import EXPECTED_SPECIAL_TOKENS, load_tokenizer_config
from atomllm.tokenizer.evaluation import REQUIRED_SUITE_ORDER
from atomllm.tokenizer.training import _build_tokenizer


TOKENIZER_CONFIG_PATH = Path("configs/tokenizer/smoke-32k.yaml")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, sort_keys=True)}\n",
        encoding="utf-8",
    )


def complete(directory: Path, payload_name: str) -> None:
    payload_path = directory / payload_name
    (directory / "COMPLETED").write_text(
        f"{sha256(payload_path)}  {payload_name}\n",
        encoding="utf-8",
    )


def build_synthetic_artifacts(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    config = load_tokenizer_config(TOKENIZER_CONFIG_PATH)
    config = replace(
        config,
        algorithm=replace(config.algorithm, vocab_size=512, min_frequency=1),
    )
    tokenizer, trainer = _build_tokenizer(config)
    tokenizer.train_from_iterator(
        [
            "合成中文文本 synthetic English text 12345",
            "繁體中文 日本語 code = value + 1",
        ],
        trainer=trainer,
        length=2,
    )
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    monkeypatch.setattr(audit, "TOKENIZER_VOCAB_SIZE", actual_vocab_size)

    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    special_tokens = [
        {"id": token_id, "token": token, "purpose": purpose}
        for token_id, token, purpose in EXPECTED_SPECIAL_TOKENS
    ]
    tokenizer_manifest = {
        "artifact_id": "tokenizer-synthetic-smoke-v1-0123456789ab",
        "status": "smoke",
        "training_eligible": False,
        "vocab_size": actual_vocab_size,
        "model_max_length": 8192,
        "special_tokens": special_tokens,
        "training_data": {
            "data_version_id": "data-synthetic-v1-0123456789ab",
            "document_count": 2,
            "sha256": "0" * 64,
        },
        "probe_results": {"synthetic": {"roundtrip": True, "unknown_count": 0}},
        "files": {
            "tokenizer.json": {
                "bytes": tokenizer_path.stat().st_size,
                "sha256": sha256(tokenizer_path),
            }
        },
    }
    tokenizer_manifest_path = tokenizer_dir / "manifest.json"
    write_json(tokenizer_manifest_path, tokenizer_manifest)
    complete(tokenizer_dir, "manifest.json")

    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir()
    suites = {
        name: {"characters_per_token": 1.5}
        for name in (*REQUIRED_SUITE_ORDER, "special_tokens")
    }
    suites["ja"]["characters_per_token"] = 0.75
    evaluation_report = {
        "evaluation_id": "tokenizer-eval-synthetic-v1-0123456789ab",
        "tokenizer_artifact_id": tokenizer_manifest["artifact_id"],
        "tokenizer_manifest_sha256": sha256(tokenizer_manifest_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "vocab_size": actual_vocab_size,
        "summary": {
            "unknown_count": 0,
            "roundtrip_failures": 0,
            "special_token_atomic_failures": 0,
            "all_correctness_checks_passed": True,
        },
        "suites": suites,
    }
    write_json(evaluation_dir / "report.json", evaluation_report)
    complete(evaluation_dir, "report.json")
    return tokenizer_dir, evaluation_dir


def test_builds_idempotent_stage_two_smoke_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_dir, evaluation_dir = build_synthetic_artifacts(tmp_path, monkeypatch)
    output_dir = tmp_path / "output"

    first = build_tokenizer_smoke_version(
        tokenizer_dir,
        evaluation_dir,
        output_dir,
    )
    second = build_tokenizer_smoke_version(
        tokenizer_dir,
        evaluation_dir,
        output_dir,
    )

    assert second == first
    assert first["status"] == "smoke_validated"
    assert first["stage3_interface_eligible"] is True
    assert first["formal_pretraining_eligible"] is False
    assert first["vocabulary_frozen"] is False
    assert all(first["audit"].values())
    assert (output_dir / "manifest.sha256").is_file()


def test_rejects_broken_evaluation_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_dir, evaluation_dir = build_synthetic_artifacts(tmp_path, monkeypatch)
    report_path = evaluation_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["tokenizer_sha256"] = "f" * 64
    write_json(report_path, report)
    complete(evaluation_dir, "report.json")

    with pytest.raises(TokenizerAuditError, match="different tokenizer.json"):
        build_tokenizer_smoke_version(
            tokenizer_dir,
            evaluation_dir,
            tmp_path / "output",
        )


def test_rejects_modified_tokenizer_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_dir, evaluation_dir = build_synthetic_artifacts(tmp_path, monkeypatch)
    with (tokenizer_dir / "tokenizer.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(TokenizerAuditError, match="SHA-256 mismatch"):
        build_tokenizer_smoke_version(
            tokenizer_dir,
            evaluation_dir,
            tmp_path / "output",
        )


def test_rejects_modified_existing_version_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_dir, evaluation_dir = build_synthetic_artifacts(tmp_path, monkeypatch)
    output_dir = tmp_path / "output"
    build_tokenizer_smoke_version(tokenizer_dir, evaluation_dir, output_dir)
    (output_dir / "manifest.sha256").write_text("invalid\n", encoding="utf-8")

    with pytest.raises(TokenizerAuditError, match="checksum is invalid"):
        build_tokenizer_smoke_version(tokenizer_dir, evaluation_dir, output_dir)
