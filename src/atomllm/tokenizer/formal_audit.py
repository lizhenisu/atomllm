"""Freeze a formally evaluated AtomLLM tokenizer release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from atomllm.tokenizer.config import (
    EXPECTED_SPECIAL_TOKENS,
    TOKENIZER_MODEL_MAX_LENGTH,
    TOKENIZER_VOCAB_SIZE,
)
from atomllm.tokenizer.evaluation import verify_tokenizer_directory


DEFAULT_TOKENIZER_DIR = Path("artifacts/tokenizers/atom-tokenizer-formal-v4")
DEFAULT_SNAPSHOT_DIR = Path("artifacts/tokenizer-snapshots/formal-70g-v4")
DEFAULT_HELDOUT_DIR = Path(
    "artifacts/tokenizer-evaluations/atom-tokenizer-formal-heldout-eval-v1"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizer-versions/atom-tokenizer-formal-v1")
MIN_LANGUAGE_CHARACTERS_PER_TOKEN = {
    "zh-Hans": 1.30,
    "en": 3.00,
    "zh-Hant": 1.30,
    "ja": 1.00,
}
MIN_CONTENT_CHARACTERS_PER_TOKEN = {
    "general": 1.50,
    "encyclopedia": 1.20,
    "code": 2.00,
    "math": 2.00,
    "science": 2.00,
}


class FormalTokenizerAuditError(RuntimeError):
    """Raised when a tokenizer cannot be frozen as a formal release."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalTokenizerAuditError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise FormalTokenizerAuditError(f"{label} must be an object")
    return value


def _completed(directory: Path, filename: str) -> tuple[dict[str, Any], str]:
    payload, marker = directory / filename, directory / "COMPLETED"
    if not payload.is_file() or not marker.is_file():
        raise FormalTokenizerAuditError(f"{directory.name} is incomplete")
    digest = _sha256(payload)
    if marker.read_text(encoding="utf-8") != f"{digest}  {filename}\n":
        raise FormalTokenizerAuditError(f"{directory.name} COMPLETED marker is invalid")
    return _json(payload, directory.name), digest


def _canonical(value: dict[str, Any], pretty: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def build_formal_tokenizer_version(
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    heldout_dir: str | Path = DEFAULT_HELDOUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    tokenizer_path, snapshot_path = Path(tokenizer_dir), Path(snapshot_dir)
    heldout_path, output_path = Path(heldout_dir), Path(output_dir)
    try:
        tokenizer, tokenizer_manifest, tokenizer_manifest_path = (
            verify_tokenizer_directory(tokenizer_path)
        )
    except Exception as error:
        raise FormalTokenizerAuditError(
            "tokenizer artifact verification failed"
        ) from error
    if (
        tokenizer_manifest.get("status") != "release"
        or tokenizer_manifest.get("training_eligible") is not True
    ):
        raise FormalTokenizerAuditError("tokenizer is not a release artifact")
    if (
        tokenizer.get_vocab_size(with_added_tokens=True) != TOKENIZER_VOCAB_SIZE
        or tokenizer_manifest.get("model_max_length") != TOKENIZER_MODEL_MAX_LENGTH
    ):
        raise FormalTokenizerAuditError("tokenizer model contract is invalid")
    for token_id, token, purpose in EXPECTED_SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) != token_id or tokenizer.encode(
            token, add_special_tokens=False
        ).ids != [token_id]:
            raise FormalTokenizerAuditError(f"special token contract failed: {token}")
    snapshot, snapshot_sha = _completed(snapshot_path, "manifest.json")
    heldout, heldout_sha = _completed(heldout_path, "report.json")
    training = tokenizer_manifest.get("training_data")
    snapshot_data = snapshot.get("snapshot")
    if (
        not isinstance(training, dict)
        or not isinstance(snapshot_data, dict)
        or training.get("data_version_id") != snapshot.get("data_version_id")
        or training.get("sha256") != snapshot_data.get("sha256")
    ):
        raise FormalTokenizerAuditError("tokenizer does not match formal snapshot")
    tokenizer_sha = _sha256(tokenizer_path / "tokenizer.json")
    manifest_sha = _sha256(tokenizer_manifest_path)
    heldout_summary = heldout.get("summary")
    if (
        not isinstance(heldout_summary, dict)
        or heldout.get("tokenizer_artifact_id") != tokenizer_manifest.get("artifact_id")
        or heldout.get("evaluation_scope") != "full_validation"
        or heldout_summary.get("document_count")
        != heldout.get("validation_input_document_count")
        or heldout_summary.get("unknown_count") != 0
        or heldout_summary.get("roundtrip_failures") != 0
    ):
        raise FormalTokenizerAuditError("held-out correctness gate failed")
    heldout_languages = heldout.get("by_language")
    heldout_contents = heldout.get("by_content_type")
    if not isinstance(heldout_languages, dict) or not isinstance(
        heldout_contents, dict
    ):
        raise FormalTokenizerAuditError("held-out distribution report is invalid")
    for name, minimum in MIN_LANGUAGE_CHARACTERS_PER_TOKEN.items():
        result = heldout_languages.get(name)
        if (
            not isinstance(result, dict)
            or result.get("document_count", 0) <= 0
            or result.get("characters_per_token", 0) < minimum
        ):
            raise FormalTokenizerAuditError(
                f"held-out language compression gate failed: {name}"
            )
    for name, minimum in MIN_CONTENT_CHARACTERS_PER_TOKEN.items():
        result = heldout_contents.get(name)
        if (
            not isinstance(result, dict)
            or result.get("document_count", 0) <= 0
            or result.get("characters_per_token", 0) < minimum
        ):
            raise FormalTokenizerAuditError(
                f"held-out content compression gate failed: {name}"
            )
    payload = {
        "schema_version": 1,
        "name": "atom-tokenizer-formal-v1",
        "status": "release_validated",
        "formal_pretraining_eligible": True,
        "vocabulary_frozen": True,
        "contract": {
            "vocab_size": TOKENIZER_VOCAB_SIZE,
            "model_max_length": TOKENIZER_MODEL_MAX_LENGTH,
            "special_tokens": [
                {"id": i, "token": t, "purpose": p}
                for i, t, p in EXPECTED_SPECIAL_TOKENS
            ],
        },
        "lineage": {
            "tokenizer": {
                "artifact_id": tokenizer_manifest["artifact_id"],
                "manifest_sha256": manifest_sha,
                "tokenizer_sha256": tokenizer_sha,
            },
            "snapshot": {
                "data_version_id": snapshot["data_version_id"],
                "manifest_sha256": snapshot_sha,
            },
            "heldout_evaluation": {
                "evaluation_id": heldout["evaluation_id"],
                "report_sha256": heldout_sha,
            },
        },
        "audit": {
            "tokenizer_files_verified": True,
            "special_token_contract_verified": True,
            "snapshot_lineage_verified": True,
            "heldout_gates_verified": True,
        },
    }
    identity = hashlib.sha256(_canonical(payload, False).encode("utf-8")).hexdigest()
    manifest = {
        **payload,
        "tokenizer_version_id": f"tokenizer-version-atom-tokenizer-formal-v1-{identity[:12]}",
        "identity_sha256": identity,
    }
    serialized = f"{_canonical(manifest, True)}\n"
    if output_path.exists():
        existing, _ = _completed(output_path, "manifest.json")
        if existing == manifest:
            return manifest
        raise FormalTokenizerAuditError("existing formal tokenizer version differs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(serialized, encoding="utf-8")
        (temporary / "COMPLETED").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary, output_path)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit and freeze the formal AtomLLM tokenizer."
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--heldout-dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_formal_tokenizer_version(
        args.tokenizer_dir,
        args.snapshot_dir,
        args.heldout_dir,
        args.output_dir,
    )
    print(f"Formal tokenizer audit complete: {manifest['tokenizer_version_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
