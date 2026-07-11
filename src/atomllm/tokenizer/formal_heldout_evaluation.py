"""Low-memory correctness and compression evaluation on formal validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from atomllm.data.schema import CanonicalDocument
from atomllm.tokenizer.evaluation import (
    TokenizerEvaluationError,
    verify_tokenizer_directory,
)


HELDOUT_EVALUATION_SCHEMA_VERSION = 1
DEFAULT_TOKENIZER_DIR = Path("artifacts/tokenizers/atom-tokenizer-formal-v4")
DEFAULT_SNAPSHOT_DIR = Path("artifacts/tokenizer-snapshots/formal-70g-v4")
DEFAULT_SPLIT_DIR = Path("artifacts/data/formal-70g/split-v1")
DEFAULT_AUDIT_DIR = Path("artifacts/data/formal-70g/audit-v1")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/tokenizer-evaluations/atom-tokenizer-formal-heldout-eval-v1"
)


class FormalHeldoutEvaluationError(RuntimeError):
    """Raised when held-out tokenizer evaluation is incomplete or invalid."""


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
        raise FormalHeldoutEvaluationError(f"cannot read {context}: {path}") from error
    if not isinstance(value, dict):
        raise FormalHeldoutEvaluationError(f"{context} must be a JSON object")
    return value


def _canonical_json(value: dict[str, Any], *, pretty: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _resolve(root: Path, path: str | Path, field_name: str) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not resolved.is_relative_to(root):
        raise FormalHeldoutEvaluationError(
            f"{field_name} resolves outside project root"
        )
    return resolved


def _validate_lineage(
    tokenizer_dir: Path,
    snapshot_dir: Path,
    split_dir: Path,
    audit_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    try:
        tokenizer, tokenizer_manifest, tokenizer_manifest_path = (
            verify_tokenizer_directory(tokenizer_dir)
        )
    except TokenizerEvaluationError as error:
        raise FormalHeldoutEvaluationError(str(error)) from error
    if (
        tokenizer_manifest.get("status") != "release"
        or tokenizer_manifest.get("training_eligible") is not True
    ):
        raise FormalHeldoutEvaluationError(
            "tokenizer is not a release training artifact"
        )
    snapshot_manifest_path = snapshot_dir / "manifest.json"
    split_manifest_path = split_dir / "manifest.json"
    audit_manifest_path = audit_dir / "manifest.json"
    for path in (snapshot_manifest_path, split_manifest_path, audit_manifest_path):
        if not path.is_file():
            raise FormalHeldoutEvaluationError(f"required manifest is missing: {path}")
    snapshot = _read_json(snapshot_manifest_path, "tokenizer snapshot manifest")
    split = _read_json(split_manifest_path, "formal split manifest")
    audit = _read_json(audit_manifest_path, "formal audit manifest")
    snapshot_data = snapshot.get("snapshot")
    training_data = tokenizer_manifest.get("training_data")
    if not isinstance(snapshot_data, dict) or not isinstance(training_data, dict):
        raise FormalHeldoutEvaluationError("tokenizer snapshot lineage is invalid")
    expected_training = {
        "data_version_id": snapshot.get("data_version_id"),
        "document_count": snapshot_data.get("document_count"),
        "sha256": snapshot_data.get("sha256"),
        "split": "train",
    }
    if training_data != expected_training:
        raise FormalHeldoutEvaluationError(
            "tokenizer training data does not match snapshot"
        )
    checks = audit.get("checks")
    if (
        audit.get("training_eligible") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise FormalHeldoutEvaluationError("formal audit is not fully passed")
    provenance = audit.get("provenance")
    split_sha256 = _sha256(split_manifest_path)
    if not isinstance(provenance, dict) or provenance.get("split") != split_sha256:
        raise FormalHeldoutEvaluationError("formal audit does not match split manifest")
    if split.get("training_eligible") is not True:
        raise FormalHeldoutEvaluationError("formal split is not training eligible")
    return tokenizer, tokenizer_manifest, split, audit, _sha256(tokenizer_manifest_path)


def _empty_metrics() -> dict[str, int]:
    return {
        "document_count": 0,
        "character_count": 0,
        "utf8_bytes": 0,
        "token_count": 0,
        "unknown_count": 0,
        "roundtrip_failures": 0,
    }


def _report_metrics(metrics: dict[str, int]) -> dict[str, int | float]:
    tokens = metrics["token_count"]
    if tokens == 0:
        raise FormalHeldoutEvaluationError("held-out selection has no tokens")
    return {
        **metrics,
        "characters_per_token": round(metrics["character_count"] / tokens, 6),
        "bytes_per_token": round(metrics["utf8_bytes"] / tokens, 6),
        "unknown_rate": round(metrics["unknown_count"] / tokens, 8),
    }


def evaluate_formal_heldout(
    *,
    project_root: str | Path = ".",
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Evaluate a release tokenizer against the complete validation split."""
    root = Path(project_root).resolve()
    tokenizer_path = _resolve(root, tokenizer_dir, "tokenizer_dir")
    snapshot_path = _resolve(root, snapshot_dir, "snapshot_dir")
    split_path = _resolve(root, split_dir, "split_dir")
    audit_path = _resolve(root, audit_dir, "audit_dir")
    output_path = _resolve(root, output_dir, "output_dir")
    (
        tokenizer,
        tokenizer_manifest,
        split_manifest,
        audit_manifest,
        tokenizer_manifest_sha256,
    ) = _validate_lineage(tokenizer_path, snapshot_path, split_path, audit_path)
    identity = {
        "schema_version": HELDOUT_EVALUATION_SCHEMA_VERSION,
        "evaluation_version": "formal-tokenizer-heldout-eval-v1",
        "evaluation_scope": "full_validation",
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "split_manifest_sha256": _sha256(split_path / "manifest.json"),
        "audit_manifest_sha256": _sha256(audit_path / "manifest.json"),
    }
    report_path = output_path / "report.json"
    completed_path = output_path / "COMPLETED"
    if output_path.exists():
        if not report_path.is_file() or not completed_path.is_file():
            raise FormalHeldoutEvaluationError(
                "existing held-out evaluation is incomplete"
            )
        report = _read_json(report_path, "held-out evaluation report")
        if all(report.get(key) == value for key, value in identity.items()) and (
            completed_path.read_text(encoding="utf-8")
            == f"{_sha256(report_path)}  report.json\n"
        ):
            return report
        raise FormalHeldoutEvaluationError(
            "existing held-out evaluation uses different input"
        )

    shards = split_manifest.get("shards")
    if not isinstance(shards, dict) or not isinstance(shards.get("validation"), list):
        raise FormalHeldoutEvaluationError("formal split validation shards are invalid")
    unknown_id = tokenizer.token_to_id("<unk>")
    if unknown_id is None:
        raise FormalHeldoutEvaluationError("tokenizer does not define <unk>")
    total = _empty_metrics()
    by_language: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    by_content: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    by_stratum: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    input_documents = 0
    expected_documents = split_manifest.get("splits", {}).get("validation")
    if type(expected_documents) is not int or expected_documents <= 0:
        raise FormalHeldoutEvaluationError(
            "formal split has invalid validation document count"
        )
    started_at = time.monotonic()
    print(
        f"[tokenizer-heldout] start documents={expected_documents}",
        file=sys.stderr,
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        for shard in shards["validation"]:
            if not isinstance(shard, dict):
                raise FormalHeldoutEvaluationError(
                    "validation shard metadata is invalid"
                )
            name = shard.get("name")
            if not isinstance(name, str):
                raise FormalHeldoutEvaluationError("validation shard name is invalid")
            shard_path = split_path / "validation" / "shards" / name
            digest = hashlib.sha256()
            lines = 0
            with shard_path.open("rb") as handle:
                for raw_line in handle:
                    digest.update(raw_line)
                    lines += 1
                    document = CanonicalDocument.from_json_line(
                        raw_line.decode("utf-8")
                    )
                    input_documents += 1
                    expected = unicodedata.normalize("NFC", document.text)
                    encoding = tokenizer.encode(expected, add_special_tokens=False)
                    decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
                    metrics = (
                        total,
                        by_language[document.language],
                        by_content[document.content_type],
                        by_stratum[f"{document.language}|{document.content_type}"],
                    )
                    for value in metrics:
                        value["document_count"] += 1
                        value["character_count"] += len(expected)
                        value["utf8_bytes"] += len(expected.encode("utf-8"))
                        value["token_count"] += len(encoding.ids)
                        value["unknown_count"] += encoding.ids.count(unknown_id)
                        value["roundtrip_failures"] += int(decoded != expected)
                    if input_documents % 10_000 == 0:
                        elapsed = time.monotonic() - started_at
                        print(
                            "[tokenizer-heldout] progress "
                            f"documents={input_documents}/{expected_documents} "
                            f"rate={input_documents / elapsed:.1f}docs/s",
                            file=sys.stderr,
                            flush=True,
                        )
            if digest.hexdigest() != shard.get("sha256") or lines != shard.get(
                "record_count"
            ):
                raise FormalHeldoutEvaluationError(
                    f"validation shard integrity mismatch: {name}"
                )
        if input_documents != expected_documents:
            raise FormalHeldoutEvaluationError(
                "full validation document count does not match split manifest"
            )
        summary = _report_metrics(total)
        if summary["unknown_count"] or summary["roundtrip_failures"]:
            raise FormalHeldoutEvaluationError("held-out correctness evaluation failed")
        report_payload = {
            **identity,
            "tokenizer_artifact_id": tokenizer_manifest.get("artifact_id"),
            "validation_input_document_count": input_documents,
            "summary": summary,
            "by_language": {
                key: _report_metrics(value)
                for key, value in sorted(by_language.items())
            },
            "by_content_type": {
                key: _report_metrics(value) for key, value in sorted(by_content.items())
            },
            "by_language_content_type": {
                key: _report_metrics(value) for key, value in sorted(by_stratum.items())
            },
        }
        digest = hashlib.sha256(
            _canonical_json(report_payload, pretty=False).encode("utf-8")
        ).hexdigest()
        report = {
            **report_payload,
            "evaluation_id": f"tokenizer-heldout-eval-{digest[:12]}",
            "identity_sha256": digest,
        }
        temporary_report = temporary_dir / "report.json"
        temporary_report.write_text(
            f"{_canonical_json(report, pretty=True)}\n", encoding="utf-8"
        )
        (temporary_dir / "COMPLETED").write_text(
            f"{_sha256(temporary_report)}  report.json\n", encoding="utf-8"
        )
        os.replace(temporary_dir, output_path)
        return report
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a release tokenizer on formal validation data."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = evaluate_formal_heldout(
        project_root=args.project_root,
        tokenizer_dir=args.tokenizer_dir,
        snapshot_dir=args.snapshot_dir,
        split_dir=args.split_dir,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
    )
    print(
        "Formal held-out tokenizer evaluation complete: "
        f"documents={report['summary']['document_count']}, "
        f"unknowns={report['summary']['unknown_count']}, "
        f"roundtrip_failures={report['summary']['roundtrip_failures']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
