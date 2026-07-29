"""Evaluate a public tokenizer on rows reserved from each complete source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from atomllm.data.schema import CanonicalDocument
from atomllm.tokenizer.evaluation import (
    TokenizerEvaluationError,
    verify_tokenizer_directory,
)
from atomllm.tokenizer.public_snapshot import _heldout_score


class PublicHeldoutEvaluationError(RuntimeError):
    """Raised when public held-out tokenizer evaluation is not trustworthy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicHeldoutEvaluationError(
            f"cannot read JSON artifact: {path}"
        ) from error
    if not isinstance(value, dict):
        raise PublicHeldoutEvaluationError(f"JSON artifact must be an object: {path}")
    return value


def _resolve(root: Path, path: Path, field: str) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root):
        raise PublicHeldoutEvaluationError(f"{field} resolves outside project root")
    return resolved


def _empty_metrics() -> dict[str, int]:
    return {
        "document_count": 0,
        "character_count": 0,
        "utf8_bytes": 0,
        "token_count": 0,
        "unknown_count": 0,
        "roundtrip_failures": 0,
    }


def _reported(metrics: dict[str, int]) -> dict[str, int | float]:
    tokens = metrics["token_count"]
    if tokens <= 0:
        raise PublicHeldoutEvaluationError("held-out metric group contains no tokens")
    return {
        **metrics,
        "characters_per_token": round(metrics["character_count"] / tokens, 6),
        "bytes_per_token": round(metrics["utf8_bytes"] / tokens, 6),
        "unknown_rate": round(metrics["unknown_count"] / tokens, 10),
    }


def evaluate(
    *,
    tokenizer_dir: Path,
    snapshot_dir: Path,
    output_dir: Path,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    root = project_root.resolve()
    tokenizer_path = _resolve(root, tokenizer_dir, "tokenizer_dir")
    snapshot_path = _resolve(root, snapshot_dir, "snapshot_dir")
    output_path = _resolve(root, output_dir, "output_dir")
    try:
        tokenizer, tokenizer_manifest, tokenizer_manifest_path = (
            verify_tokenizer_directory(tokenizer_path)
        )
    except TokenizerEvaluationError as error:
        raise PublicHeldoutEvaluationError(str(error)) from error
    if tokenizer_manifest.get("training_eligible") is not True:
        raise PublicHeldoutEvaluationError("tokenizer is not training eligible")
    snapshot_manifest_path = snapshot_path / "manifest.json"
    snapshot_completed = snapshot_path / "COMPLETED"
    heldout_path = snapshot_path / "heldout.jsonl"
    if not all(
        path.is_file()
        for path in (snapshot_manifest_path, snapshot_completed, heldout_path)
    ):
        raise PublicHeldoutEvaluationError("tokenizer snapshot is incomplete")
    if snapshot_completed.read_text(encoding="utf-8") != (
        f"{_sha256(snapshot_manifest_path)}  manifest.json\n"
    ):
        raise PublicHeldoutEvaluationError("tokenizer snapshot marker is invalid")
    snapshot = _read_json(snapshot_manifest_path)
    heldout = snapshot.get("heldout")
    documents = snapshot.get("documents")
    if not isinstance(heldout, dict) or not isinstance(documents, dict):
        raise PublicHeldoutEvaluationError("snapshot document metadata is invalid")
    if heldout.get("disjoint_from_training_by_construction") is not True:
        raise PublicHeldoutEvaluationError("held-out disjointness is not declared")
    if snapshot.get("heldout_selection_method") != ("source-lowest-sha256-reserved-v3"):
        raise PublicHeldoutEvaluationError("held-out selection method is invalid")
    if heldout_path.stat().st_size != heldout.get("size_bytes"):
        raise PublicHeldoutEvaluationError("held-out file size mismatch")
    if _sha256(heldout_path) != heldout.get("sha256"):
        raise PublicHeldoutEvaluationError("held-out file hash mismatch")
    expected_training = {
        "data_version_id": snapshot.get("data_version_id"),
        "split": "train",
        "document_count": snapshot.get("document_count"),
        "sha256": documents.get("sha256"),
    }
    if tokenizer_manifest.get("training_data") != expected_training:
        raise PublicHeldoutEvaluationError(
            "tokenizer training lineage does not match snapshot"
        )
    sample_ratio = snapshot.get("sample_ratio")
    seed = snapshot.get("selection_seed")
    if type(sample_ratio) not in {int, float} or not 0 < float(sample_ratio) < 1:
        raise PublicHeldoutEvaluationError("snapshot sample ratio is invalid")
    if type(seed) is not int or seed < 0:
        raise PublicHeldoutEvaluationError("snapshot selection seed is invalid")
    identity = {
        "schema_version": 1,
        "evaluation_version": "public-tokenizer-heldout-v1",
        "tokenizer_manifest_sha256": _sha256(tokenizer_manifest_path),
        "snapshot_manifest_sha256": _sha256(snapshot_manifest_path),
        "heldout_sha256": heldout["sha256"],
    }
    report_path = output_path / "report.json"
    completed_path = output_path / "COMPLETED"
    if output_path.exists():
        if report_path.is_file() and completed_path.is_file():
            existing = _read_json(report_path)
            if all(existing.get(key) == value for key, value in identity.items()) and (
                completed_path.read_text(encoding="utf-8")
                == f"{_sha256(report_path)}  report.json\n"
            ):
                return existing
        raise PublicHeldoutEvaluationError(
            "existing held-out evaluation is incompatible"
        )

    unknown_id = tokenizer.token_to_id("<unk>")
    if unknown_id is None:
        raise PublicHeldoutEvaluationError("tokenizer has no <unk> token")
    total = _empty_metrics()
    by_language: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    by_content: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    by_source: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    source_documents: Counter[str] = Counter()
    previous_source = ""
    previous_score = -1
    encode_seconds = 0.0
    decode_seconds = 0.0
    with heldout_path.open(encoding="utf-8") as handle:
        for line in handle:
            document = CanonicalDocument.from_json_line(line)
            score = _heldout_score(document.document_id, seed + 1)
            if document.source_id < previous_source or (
                document.source_id == previous_source and score < previous_score
            ):
                raise PublicHeldoutEvaluationError(
                    "held-out rows are not ordered by source and SHA-256 score"
                )
            previous_source = document.source_id
            previous_score = score
            expected = unicodedata.normalize("NFC", document.text)
            started = time.perf_counter()
            encoding = tokenizer.encode(expected, add_special_tokens=False)
            encode_seconds += time.perf_counter() - started
            started = time.perf_counter()
            decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
            decode_seconds += time.perf_counter() - started
            source_documents[document.source_id] += 1
            groups = (
                total,
                by_language[document.language],
                by_content[document.content_type],
                by_source[document.source_id],
            )
            for metrics in groups:
                metrics["document_count"] += 1
                metrics["character_count"] += len(expected)
                metrics["utf8_bytes"] += len(expected.encode("utf-8"))
                metrics["token_count"] += len(encoding.ids)
                metrics["unknown_count"] += encoding.ids.count(unknown_id)
                metrics["roundtrip_failures"] += int(decoded != expected)
    if total["document_count"] != heldout.get("document_count"):
        raise PublicHeldoutEvaluationError("held-out document count mismatch")
    if dict(sorted(source_documents.items())) != heldout.get("source_documents"):
        raise PublicHeldoutEvaluationError("held-out source counts mismatch")
    expected_per_source = snapshot.get("heldout_documents_per_source")
    if type(expected_per_source) is not int or any(
        count != expected_per_source for count in source_documents.values()
    ):
        raise PublicHeldoutEvaluationError(
            "held-out must reserve the configured count from every source"
        )
    if total["unknown_count"] or total["roundtrip_failures"]:
        raise PublicHeldoutEvaluationError("held-out tokenizer correctness failed")
    summary = _reported(total)
    summary["encode_tokens_per_second"] = round(
        total["token_count"] / encode_seconds, 2
    )
    summary["decode_tokens_per_second"] = round(
        total["token_count"] / decode_seconds, 2
    )
    report = {
        **identity,
        "tokenizer_artifact_id": tokenizer_manifest.get("artifact_id"),
        "vocab_size": tokenizer_manifest.get("vocab_size"),
        "summary": summary,
        "by_language": {
            key: _reported(value) for key, value in sorted(by_language.items())
        },
        "by_content_type": {
            key: _reported(value) for key, value in sorted(by_content.items())
        },
        "by_source": {
            key: _reported(value) for key, value in sorted(by_source.items())
        },
        "checks": {
            "full_source_lowest_sha256_reserved": True,
            "roundtrip_failures": 0,
            "unknown_count": 0,
            "model_external_capability": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        temporary_report = temporary_dir / "report.json"
        temporary_report.write_text(
            json.dumps(
                report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "COMPLETED").write_text(
            f"{_sha256(temporary_report)}  report.json\n", encoding="utf-8"
        )
        os.replace(temporary_dir, output_path)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = evaluate(
        tokenizer_dir=args.tokenizer_dir,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
