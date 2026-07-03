"""Machine-verifiable stage two tokenizer smoke audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from atomllm.tokenizer.config import (
    EXPECTED_SPECIAL_TOKENS,
    TOKENIZER_MODEL_MAX_LENGTH,
    TOKENIZER_VOCAB_SIZE,
)
from atomllm.tokenizer.evaluation import REQUIRED_SUITE_ORDER


TOKENIZER_AUDIT_SCHEMA_VERSION = 1
TOKENIZER_SMOKE_VERSION_NAME = "atom-tokenizer-smoke-v1"


class TokenizerAuditError(RuntimeError):
    """Raised when tokenizer or evaluation lineage fails the stage two audit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerAuditError(f"cannot read {context}: {path.name}") from error
    if not isinstance(value, dict):
        raise TokenizerAuditError(f"{context} must be a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise TokenizerAuditError(message)


def _verify_completed(directory: Path, payload_name: str) -> tuple[Path, str]:
    payload_path = directory / payload_name
    completed_path = directory / "COMPLETED"
    if not payload_path.is_file() or not completed_path.is_file():
        raise TokenizerAuditError(f"{directory.name} is incomplete")
    payload_sha256 = _sha256(payload_path)
    if completed_path.read_text(encoding="utf-8") != (
        f"{payload_sha256}  {payload_name}\n"
    ):
        raise TokenizerAuditError(f"{directory.name} COMPLETED marker is invalid")
    return payload_path, payload_sha256


def _verify_tokenizer_artifact(
    directory: Path,
) -> tuple[dict[str, Any], str, str]:
    manifest_path, manifest_sha256 = _verify_completed(directory, "manifest.json")
    manifest = _read_json(manifest_path, "tokenizer manifest")
    _require_equal(manifest.get("status"), "smoke", "tokenizer status must be smoke")
    _require_equal(
        manifest.get("training_eligible"),
        False,
        "smoke tokenizer must not be training eligible",
    )
    _require_equal(
        manifest.get("vocab_size"),
        TOKENIZER_VOCAB_SIZE,
        "tokenizer vocabulary size does not match the model contract",
    )
    _require_equal(
        manifest.get("model_max_length"),
        TOKENIZER_MODEL_MAX_LENGTH,
        "tokenizer model_max_length does not match the model contract",
    )

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TokenizerAuditError("tokenizer manifest has invalid files")
    for name, metadata in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(metadata, dict)
        ):
            raise TokenizerAuditError("tokenizer file metadata is invalid")
        path = directory / name
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            raise TokenizerAuditError(f"tokenizer file SHA-256 mismatch: {name}")

    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise TokenizerAuditError("tokenizer.json is missing")
    tokenizer_sha256 = _sha256(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    _require_equal(
        tokenizer.get_vocab_size(with_added_tokens=True),
        TOKENIZER_VOCAB_SIZE,
        "reloaded tokenizer vocabulary size is invalid",
    )
    expected_special_tokens = [
        {"id": token_id, "token": token, "purpose": purpose}
        for token_id, token, purpose in EXPECTED_SPECIAL_TOKENS
    ]
    _require_equal(
        manifest.get("special_tokens"),
        expected_special_tokens,
        "tokenizer manifest special-token protocol is invalid",
    )
    for token_id, token, _ in EXPECTED_SPECIAL_TOKENS:
        _require_equal(
            tokenizer.token_to_id(token),
            token_id,
            f"reloaded tokenizer special-token ID mismatch: {token}",
        )
        _require_equal(
            tokenizer.encode(token, add_special_tokens=False).ids,
            [token_id],
            f"reloaded tokenizer special token is not atomic: {token}",
        )

    probe_results = manifest.get("probe_results")
    if not isinstance(probe_results, dict) or not probe_results:
        raise TokenizerAuditError("tokenizer manifest has no probe results")
    if any(
        result.get("roundtrip") is not True or result.get("unknown_count") != 0
        for result in probe_results.values()
        if isinstance(result, dict)
    ) or not all(isinstance(result, dict) for result in probe_results.values()):
        raise TokenizerAuditError("tokenizer training probes did not all pass")
    return manifest, manifest_sha256, tokenizer_sha256


def _verify_evaluation_artifact(
    directory: Path,
    tokenizer_manifest: Mapping[str, Any],
    tokenizer_manifest_sha256: str,
    tokenizer_sha256: str,
) -> tuple[dict[str, Any], str]:
    report_path, report_sha256 = _verify_completed(directory, "report.json")
    report = _read_json(report_path, "tokenizer evaluation report")
    _require_equal(
        report.get("tokenizer_artifact_id"),
        tokenizer_manifest.get("artifact_id"),
        "evaluation references a different tokenizer artifact",
    )
    _require_equal(
        report.get("tokenizer_manifest_sha256"),
        tokenizer_manifest_sha256,
        "evaluation references a different tokenizer manifest",
    )
    _require_equal(
        report.get("tokenizer_sha256"),
        tokenizer_sha256,
        "evaluation references a different tokenizer.json",
    )
    _require_equal(
        report.get("vocab_size"),
        TOKENIZER_VOCAB_SIZE,
        "evaluation vocabulary size is invalid",
    )
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise TokenizerAuditError("evaluation report has invalid summary")
    expected_summary = {
        "unknown_count": 0,
        "roundtrip_failures": 0,
        "special_token_atomic_failures": 0,
        "all_correctness_checks_passed": True,
    }
    for field_name, expected in expected_summary.items():
        _require_equal(
            summary.get(field_name),
            expected,
            f"evaluation correctness check failed: {field_name}",
        )
    suites = report.get("suites")
    if not isinstance(suites, dict):
        raise TokenizerAuditError("evaluation report has invalid suites")
    expected_suites = (*REQUIRED_SUITE_ORDER, "special_tokens")
    if len(suites) != len(expected_suites) or set(suites) != set(expected_suites):
        raise TokenizerAuditError(
            "evaluation suites do not match the stage two contract"
        )
    return report, report_sha256


def _canonical_json(value: Mapping[str, Any], *, pretty: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _write_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def build_tokenizer_smoke_version(
    tokenizer_dir: str | Path,
    evaluation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit tokenizer lineage and write a deterministic stage two version."""
    tokenizer_path = Path(tokenizer_dir)
    evaluation_path = Path(evaluation_dir)
    tokenizer_manifest, tokenizer_manifest_sha256, tokenizer_sha256 = (
        _verify_tokenizer_artifact(tokenizer_path)
    )
    evaluation_report, evaluation_report_sha256 = _verify_evaluation_artifact(
        evaluation_path,
        tokenizer_manifest,
        tokenizer_manifest_sha256,
        tokenizer_sha256,
    )

    japanese_compression = evaluation_report["suites"]["ja"]["characters_per_token"]
    payload = {
        "schema_version": TOKENIZER_AUDIT_SCHEMA_VERSION,
        "name": TOKENIZER_SMOKE_VERSION_NAME,
        "status": "smoke_validated",
        "stage3_interface_eligible": True,
        "formal_pretraining_eligible": False,
        "vocabulary_frozen": False,
        "formal_ineligible_reasons": [
            "training_data_is_a_single_source_smoke_sample",
            "training_data_is_not_formally_eligible",
            "japanese_compression_is_below_one_character_per_token",
            "formal_multilingual_tokenizer_has_not_been_trained",
        ],
        "contract": {
            "vocab_size": TOKENIZER_VOCAB_SIZE,
            "model_max_length": TOKENIZER_MODEL_MAX_LENGTH,
            "special_tokens": [
                {"id": token_id, "token": token, "purpose": purpose}
                for token_id, token, purpose in EXPECTED_SPECIAL_TOKENS
            ],
        },
        "lineage": {
            "tokenizer": {
                "artifact_id": tokenizer_manifest["artifact_id"],
                "manifest_sha256": tokenizer_manifest_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "training_data": tokenizer_manifest["training_data"],
            },
            "evaluation": {
                "evaluation_id": evaluation_report["evaluation_id"],
                "report_sha256": evaluation_report_sha256,
                "summary": evaluation_report["summary"],
                "japanese_characters_per_token": japanese_compression,
            },
        },
        "audit": {
            "tokenizer_manifest_verified": True,
            "tokenizer_files_verified": True,
            "tokenizer_reload_verified": True,
            "vocab_size_verified": True,
            "special_token_ids_verified": True,
            "special_token_atomicity_verified": True,
            "training_probes_verified": True,
            "evaluation_lineage_verified": True,
            "evaluation_correctness_verified": True,
            "formal_limitations_recorded": True,
        },
    }
    identity = _canonical_json(payload, pretty=False)
    identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    manifest = {
        **payload,
        "tokenizer_version_id": (
            f"tokenizer-version-{TOKENIZER_SMOKE_VERSION_NAME}-{identity_sha256[:12]}"
        ),
        "identity_sha256": identity_sha256,
    }
    serialized = f"{_canonical_json(manifest, pretty=True)}\n"
    manifest_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    checksum_path = output_path / "manifest.sha256"
    if manifest_path.exists() or checksum_path.exists():
        if not manifest_path.is_file() or not checksum_path.is_file():
            raise TokenizerAuditError("existing tokenizer smoke version is incomplete")
        if manifest_path.read_text(encoding="utf-8") != serialized:
            raise TokenizerAuditError(
                "existing tokenizer smoke version has different content"
            )
        if checksum_path.read_text(encoding="utf-8") != (
            f"{manifest_sha256}  manifest.json\n"
        ):
            raise TokenizerAuditError(
                "existing tokenizer smoke version checksum is invalid"
            )
        return manifest
    _write_atomic(manifest_path, serialized)
    _write_atomic(
        checksum_path,
        f"{manifest_sha256}  manifest.json\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and version the stage two tokenizer smoke artifacts."
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("artifacts/tokenizers/atom-tokenizer-smoke-v1"),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("artifacts/tokenizer-evaluations/atom-tokenizer-smoke-eval-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer-versions/atom-tokenizer-smoke-v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_tokenizer_smoke_version(
        args.tokenizer_dir,
        args.evaluation_dir,
        args.output_dir,
    )
    print(
        "Tokenizer smoke audit complete: "
        f"{manifest['tokenizer_version_id']}, "
        f"stage3_interface_eligible="
        f"{str(manifest['stage3_interface_eligible']).lower()}, "
        f"formal_pretraining_eligible="
        f"{str(manifest['formal_pretraining_eligible']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
