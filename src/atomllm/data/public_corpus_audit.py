"""Full-scan acceptance audit for the public tokenizer corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from atomllm.data.acquisition import (
    chinese_script_classifier_identity,
    classify_chinese_script,
)
from atomllm.data.public_tokenizer_corpus import Source, load_config
from atomllm.data.schema import CanonicalDocument


class PublicCorpusAuditError(RuntimeError):
    """Raised when the public corpus cannot be accepted for tokenizer training."""


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
        raise PublicCorpusAuditError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise PublicCorpusAuditError(f"JSON artifact must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _audit_document(document: CanonicalDocument, source: Source) -> tuple[int, bytes]:
    text_bytes = document.text.encode("utf-8")
    if not source.minimum_characters <= len(document.text) <= source.maximum_characters:
        raise PublicCorpusAuditError(
            f"document violates source length bounds: {document.document_id}"
        )
    if (
        document.language != source.language
        or document.content_type != source.content_type
    ):
        raise PublicCorpusAuditError(
            f"document source classification mismatch: {document.document_id}"
        )
    if (
        source.language == "zh-Hans"
        and classify_chinese_script(document.text) != "zh-Hans"
    ):
        raise PublicCorpusAuditError(
            f"non-Simplified Chinese document passed the corpus: {document.document_id}"
        )
    metadata = document.metadata
    if metadata.get("local_text_conversion") != "none":
        raise PublicCorpusAuditError(
            f"document used local text conversion: {document.document_id}"
        )
    if source.score_field is not None:
        score = metadata.get("upstream_quality_score")
        field = metadata.get("upstream_quality_score_field")
        if type(score) not in {int, float} or field != source.score_field:
            raise PublicCorpusAuditError(
                f"document is missing upstream quality provenance: {document.document_id}"
            )
        if float(score) < source.minimum_score:  # type: ignore[arg-type]
            raise PublicCorpusAuditError(
                f"document is below its upstream quality threshold: {document.document_id}"
            )
    return len(text_bytes), hashlib.sha256(text_bytes).digest()


_AUDIT_WORKER_SOURCES: dict[str, Source] = {}


def _initialize_audit_worker(sources: dict[str, Source]) -> None:
    global _AUDIT_WORKER_SOURCES
    _AUDIT_WORKER_SOURCES = sources


def _audit_document_worker(document: CanonicalDocument) -> tuple[int, bytes]:
    source = _AUDIT_WORKER_SOURCES.get(document.source_id)
    if source is None:
        raise PublicCorpusAuditError(
            f"audit worker received unknown source: {document.source_id}"
        )
    return _audit_document(document, source)


def _audit_batch(
    *,
    batch: list[tuple[int, CanonicalDocument, Source]],
    executor: ProcessPoolExecutor | None,
    document_ids: set[str],
    text_digests: set[bytes],
    source_documents: Counter[str],
    source_text_bytes: Counter[str],
    language_text_bytes: Counter[str],
    content_text_bytes: Counter[str],
) -> None:
    results: list[tuple[int, bytes] | None] = [None] * len(batch)
    parallel_indices = [
        index
        for index, (_, _, source) in enumerate(batch)
        if executor is not None and source.language == "zh-Hans"
    ]
    if parallel_indices:
        audited = executor.map(
            _audit_document_worker,
            (batch[index][1] for index in parallel_indices),
            chunksize=8,
        )
        for index, result in zip(parallel_indices, audited, strict=True):
            results[index] = result
    for index, (_, document, source) in enumerate(batch):
        if results[index] is None:
            results[index] = _audit_document(document, source)
    for (line_number, document, _source), result in zip(batch, results, strict=True):
        if result is None:
            raise AssertionError("audit result is missing")
        text_bytes, text_digest = result
        if document.document_id in document_ids:
            raise PublicCorpusAuditError(
                f"duplicate document_id: {document.document_id}"
            )
        if text_digest in text_digests:
            raise PublicCorpusAuditError(
                f"duplicate document text at line {line_number}"
            )
        document_ids.add(document.document_id)
        text_digests.add(text_digest)
        source_documents[document.source_id] += 1
        source_text_bytes[document.source_id] += text_bytes
        language_text_bytes[document.language] += text_bytes
        content_text_bytes[document.content_type] += text_bytes


def audit(
    *,
    corpus_dir: Path,
    output_dir: Path,
    project_root: Path = Path("."),
    workers: int = 1,
) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    if not 1 <= workers <= cpu_count:
        raise PublicCorpusAuditError(f"workers must be in [1, {cpu_count}]")
    root = project_root.resolve()
    corpus = (root / corpus_dir).resolve()
    output = (root / output_dir).resolve()
    if not corpus.is_relative_to(root) or not output.is_relative_to(root):
        raise PublicCorpusAuditError("audit paths must remain inside the project root")
    manifest_path = corpus / "manifest.json"
    completed_path = corpus / "COMPLETED"
    config_path = corpus / "config.yaml"
    documents_path = corpus / "documents.jsonl"
    if not all(
        path.is_file()
        for path in (manifest_path, completed_path, config_path, documents_path)
    ):
        raise PublicCorpusAuditError("public corpus is incomplete")
    manifest_sha = _sha256(manifest_path)
    if completed_path.read_text(encoding="utf-8") != (
        f"{manifest_sha}  manifest.json\n"
    ):
        raise PublicCorpusAuditError("public corpus COMPLETED marker is invalid")
    manifest = _read_json(manifest_path)
    config_sha = _sha256(config_path)
    if config_sha != manifest.get("config_sha256"):
        raise PublicCorpusAuditError("public corpus config snapshot hash mismatch")
    if manifest.get("synthetic_training_content") is not False:
        raise PublicCorpusAuditError("synthetic training content is forbidden")
    classifier_identity = chinese_script_classifier_identity()
    if manifest.get("chinese_script_classifier") != classifier_identity:
        raise PublicCorpusAuditError("Chinese script classifier identity mismatch")
    contract = manifest.get("language_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("local_text_conversion") != "none"
    ):
        raise PublicCorpusAuditError("local Chinese conversion is forbidden")
    if contract.get("privacy_filtering") != "none":
        raise PublicCorpusAuditError("local privacy-pattern filtering must be disabled")
    documents_metadata = manifest.get("documents")
    if not isinstance(documents_metadata, dict):
        raise PublicCorpusAuditError("public corpus document metadata is invalid")
    if documents_path.stat().st_size != documents_metadata.get("size_bytes"):
        raise PublicCorpusAuditError("public corpus document size mismatch")
    if _sha256(documents_path) != documents_metadata.get("sha256"):
        raise PublicCorpusAuditError("public corpus document hash mismatch")

    config = load_config(config_path)
    sources = {source.source_id: source for source in config.sources}
    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, list) or {
        item.get("source_id") for item in manifest_sources if isinstance(item, dict)
    } != set(sources):
        raise PublicCorpusAuditError("public corpus source registry mismatch")
    document_ids: set[str] = set()
    text_digests: set[bytes] = set()
    source_documents: Counter[str] = Counter()
    source_text_bytes: Counter[str] = Counter()
    language_text_bytes: Counter[str] = Counter()
    content_text_bytes: Counter[str] = Counter()
    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_audit_worker,
            initargs=(sources,),
        )
        if workers > 1
        else None
    )
    started = time.monotonic()
    audited_documents = 0
    next_progress = 10_000
    batch_size = max(512, workers * 128)

    def progress(count: int) -> None:
        nonlocal audited_documents, next_progress
        audited_documents += count
        if audited_documents < next_progress:
            return
        elapsed = time.monotonic() - started
        print(
            f"[public-corpus-audit] documents={audited_documents} "
            f"text_gib={sum(source_text_bytes.values()) / (1024**3):.3f} "
            f"documents_per_second={audited_documents / elapsed:.1f}",
            flush=True,
        )
        while next_progress <= audited_documents:
            next_progress += 10_000

    try:
        with documents_path.open(encoding="utf-8") as handle:
            batch: list[tuple[int, CanonicalDocument, Source]] = []
            for line_number, line in enumerate(handle, 1):
                try:
                    document = CanonicalDocument.from_json_line(line)
                except Exception as error:
                    raise PublicCorpusAuditError(
                        f"invalid canonical document at line {line_number}"
                    ) from error
                source = sources.get(document.source_id)
                if source is None:
                    raise PublicCorpusAuditError(
                        f"unknown source at line {line_number}: {document.source_id}"
                    )
                batch.append((line_number, document, source))
                if len(batch) < batch_size:
                    continue
                _audit_batch(
                    batch=batch,
                    executor=executor,
                    document_ids=document_ids,
                    text_digests=text_digests,
                    source_documents=source_documents,
                    source_text_bytes=source_text_bytes,
                    language_text_bytes=language_text_bytes,
                    content_text_bytes=content_text_bytes,
                )
                progress(len(batch))
                batch.clear()
            if batch:
                _audit_batch(
                    batch=batch,
                    executor=executor,
                    document_ids=document_ids,
                    text_digests=text_digests,
                    source_documents=source_documents,
                    source_text_bytes=source_text_bytes,
                    language_text_bytes=language_text_bytes,
                    content_text_bytes=content_text_bytes,
                )
                progress(len(batch))
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    document_count = sum(source_documents.values())
    expected_source_documents = manifest.get("source_documents")
    expected_source_bytes = manifest.get("source_text_bytes")
    if dict(source_documents) != expected_source_documents:
        raise PublicCorpusAuditError("source document totals do not match manifest")
    if dict(source_text_bytes) != expected_source_bytes:
        raise PublicCorpusAuditError("source byte totals do not match manifest")
    if dict(sorted(language_text_bytes.items())) != manifest.get("language_text_bytes"):
        raise PublicCorpusAuditError("language byte totals do not match manifest")
    if dict(sorted(content_text_bytes.items())) != manifest.get("content_text_bytes"):
        raise PublicCorpusAuditError("content byte totals do not match manifest")
    if document_count != manifest.get("document_count"):
        raise PublicCorpusAuditError("document count does not match manifest")
    for source_id, source in sources.items():
        actual = source_text_bytes[source_id]
        maximum_overshoot = source.maximum_characters * 4
        if (
            not source.target_text_bytes
            <= actual
            <= (source.target_text_bytes + maximum_overshoot)
        ):
            raise PublicCorpusAuditError(
                f"source byte quota is invalid: {source_id}={actual}"
            )
    report = {
        "schema_version": 1,
        "training_eligible": True,
        "corpus_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "documents_sha256": documents_metadata["sha256"],
        "document_count": document_count,
        "text_bytes": sum(source_text_bytes.values()),
        "workers": workers,
        "chinese_script_classifier": classifier_identity,
        "language_text_bytes": dict(sorted(language_text_bytes.items())),
        "content_text_bytes": dict(sorted(content_text_bytes.items())),
        "source_text_bytes": dict(sorted(source_text_bytes.items())),
        "checks": {
            "canonical_documents": True,
            "fixed_public_sources": True,
            "upstream_quality_thresholds": True,
            "simplified_chinese_only": True,
            "unique_document_ids": True,
            "exact_text_deduplication": True,
            "synthetic_training_content": False,
            "local_text_conversion": "none",
            "local_privacy_filtering": "none",
        },
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    (output / "COMPLETED").write_text(
        f"{_sha256(report_path)}  report.json\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    report = audit(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
