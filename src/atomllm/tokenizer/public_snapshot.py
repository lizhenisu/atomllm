"""Create deterministic capacity-adjustable snapshots from the public corpus."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from atomllm.data.schema import CanonicalDocument
from atomllm.tokenizer.config import EXPECTED_SPECIAL_TOKENS


class PublicTokenizerSnapshotError(RuntimeError):
    """Raised when a tokenizer snapshot would violate corpus lineage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_source(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    completed = directory / "COMPLETED"
    documents = directory / "documents.jsonl"
    if not all(path.is_file() for path in (manifest_path, completed, documents)):
        raise PublicTokenizerSnapshotError("public tokenizer corpus is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completed.read_text(encoding="utf-8") != (
        f"{_sha256(manifest_path)}  manifest.json\n"
    ):
        raise PublicTokenizerSnapshotError("public corpus COMPLETED marker is invalid")
    if manifest.get("synthetic_training_content") is not False:
        raise PublicTokenizerSnapshotError("synthetic training content is forbidden")
    contract = manifest.get("language_contract", {})
    if contract.get("local_text_conversion") != "none":
        raise PublicTokenizerSnapshotError("local Chinese conversion is forbidden")
    if contract.get("privacy_filtering") != "none":
        raise PublicTokenizerSnapshotError(
            "public corpus must not use local privacy-pattern filtering"
        )
    metadata = manifest.get("documents", {})
    if metadata.get("size_bytes") != documents.stat().st_size or metadata.get(
        "sha256"
    ) != _sha256(documents):
        raise PublicTokenizerSnapshotError("public corpus document hash mismatch")
    return manifest


def _verify_audit(
    directory: Path,
    source_dir: Path,
    *,
    verified_documents_sha256: str | None = None,
) -> dict[str, Any]:
    report_path = directory / "report.json"
    completed_path = directory / "COMPLETED"
    if not report_path.is_file() or not completed_path.is_file():
        raise PublicTokenizerSnapshotError("public corpus audit is incomplete")
    if completed_path.read_text(encoding="utf-8") != (
        f"{_sha256(report_path)}  report.json\n"
    ):
        raise PublicTokenizerSnapshotError("public corpus audit marker is invalid")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("training_eligible") is not True:
        raise PublicTokenizerSnapshotError("public corpus audit is not eligible")
    if report.get("corpus_manifest_sha256") != _sha256(source_dir / "manifest.json"):
        raise PublicTokenizerSnapshotError("public corpus audit lineage mismatch")
    documents_sha = verified_documents_sha256 or _sha256(source_dir / "documents.jsonl")
    if report.get("documents_sha256") != documents_sha:
        raise PublicTokenizerSnapshotError("public corpus audit document mismatch")
    return report


def _selected(document_id: str, ratio: float, seed: int) -> bool:
    threshold = int(ratio * (1 << 256))
    digest = hashlib.sha256(f"{seed}\0{document_id}".encode()).digest()
    return int.from_bytes(digest, "big") < threshold


def _heldout_score(document_id: str, seed: int) -> int:
    """Return the deterministic SHA-256 score used to reserve held-out rows."""
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{document_id}".encode()).digest(), "big"
    )


def _offer_heldout(
    heap: list[tuple[int, str, str, int]],
    *,
    document: CanonicalDocument,
    line: str,
    text_bytes: int,
    limit: int,
    seed: int,
) -> None:
    """Keep the lowest deterministic SHA-256 scores from one complete source."""
    score = _heldout_score(document.document_id, seed)
    entry = (-score, document.document_id, line, text_bytes)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _training_config(
    *,
    name: str,
    vocab_size: int,
    data_version_id: str,
    document_count: int,
    documents_sha256: str,
    documents_path: Path,
    tokenizer_output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "status": "release",
        "training_eligible": True,
        "model_max_length": 8192,
        "algorithm": {
            "model_type": "byte_level_bpe",
            "vocab_size": vocab_size,
            "normalization": "nfc",
            "pre_tokenizer": "byte_level",
            "decoder": "byte_level",
            "min_frequency": 2,
            "dropout": 0.0,
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": True,
            "byte_fallback": False,
            "fuse_unk": False,
            "ignore_merges": False,
            "max_token_length": 24,
        },
        "special_tokens": [
            {"id": token_id, "token": token, "purpose": purpose}
            for token_id, token, purpose in EXPECTED_SPECIAL_TOKENS
        ],
        "training_data": {
            "data_version_id": data_version_id,
            "split": "train",
            "document_count": document_count,
            "expected_sha256": documents_sha256,
            "input_path": documents_path.as_posix(),
        },
        "evaluation": {
            "roundtrip_required": True,
            "max_unknown_rate": 0.0,
            "suites": [
                "zh-Hans",
                "en",
                "code",
                "math",
                "digits",
                "whitespace",
            ],
        },
        "output_dir": tokenizer_output_dir.as_posix(),
    }


def build(
    *,
    source_dir: Path,
    audit_dir: Path,
    output_dir: Path,
    tokenizer_output_dir: Path,
    sample_ratio: float,
    candidate_vocab_sizes: tuple[int, ...] = (32000, 48000),
    heldout_documents_per_source: int = 100,
    artifact_label: str | None = None,
    seed: int = 20260718,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    if not 0 < sample_ratio <= 1:
        raise PublicTokenizerSnapshotError("sample_ratio must be in (0, 1]")
    if type(seed) is not int or seed < 0:
        raise PublicTokenizerSnapshotError("seed must be non-negative")
    if (
        not candidate_vocab_sizes
        or len(candidate_vocab_sizes) != len(set(candidate_vocab_sizes))
        or any(
            type(size) is not int or not 256 <= size <= 65535
            for size in candidate_vocab_sizes
        )
    ):
        raise PublicTokenizerSnapshotError(
            "candidate_vocab_sizes must be unique integers in [256, 65535]"
        )
    if (
        type(heldout_documents_per_source) is not int
        or heldout_documents_per_source <= 0
    ):
        raise PublicTokenizerSnapshotError(
            "heldout_documents_per_source must be a positive integer"
        )
    ratio_label = f"{round(sample_ratio * 100):03d}pct"
    artifact_label = artifact_label or ratio_label
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", artifact_label) is None:
        raise PublicTokenizerSnapshotError(
            "artifact_label must contain lowercase letters, digits, and hyphens"
        )
    root = project_root.resolve()
    source = (root / source_dir).resolve()
    audit_directory = (root / audit_dir).resolve()
    output = (root / output_dir).resolve()
    tokenizer_output = (root / tokenizer_output_dir).resolve()
    if not all(
        path.is_relative_to(root)
        for path in (source, audit_directory, output, tokenizer_output)
    ):
        raise PublicTokenizerSnapshotError("all paths must remain in project root")
    source_manifest = _verify_source(source)
    audit_report = _verify_audit(
        audit_directory,
        source,
        verified_documents_sha256=source_manifest["documents"]["sha256"],
    )
    identity = {
        "schema_version": 1,
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "source_audit_sha256": _sha256(audit_directory / "report.json"),
        "sample_ratio": sample_ratio,
        "selection_seed": seed,
        "selection_method": "sha256(seed:document_id)-threshold-v1",
        "candidate_vocab_sizes": list(candidate_vocab_sizes),
        "artifact_label": artifact_label,
        "heldout_documents_per_source": heldout_documents_per_source,
        "heldout_selection_method": "source-lowest-sha256-reserved-v3",
    }
    manifest_path = output / "manifest.json"
    completed_path = output / "COMPLETED"
    if manifest_path.is_file() and completed_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise PublicTokenizerSnapshotError("existing snapshot identity mismatch")
        if completed_path.read_text(encoding="utf-8") != (
            f"{_sha256(manifest_path)}  manifest.json\n"
        ):
            raise PublicTokenizerSnapshotError("snapshot COMPLETED marker is invalid")
        return existing
    if output.exists():
        raise PublicTokenizerSnapshotError("incomplete snapshot output already exists")
    output.mkdir(parents=True)
    documents = output / "documents.jsonl"
    temporary = output / ".documents.jsonl.tmp"
    heldout = output / "heldout.jsonl"
    heldout_temporary = output / ".heldout.jsonl.tmp"
    document_count = 0
    text_bytes = 0
    language_bytes: Counter[str] = Counter()
    content_bytes: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    heldout_documents: Counter[str] = Counter()
    heldout_text_bytes: Counter[str] = Counter()
    heldout_heaps: dict[str, list[tuple[int, str, str, int]]] = {}
    try:
        with (source / "documents.jsonl").open(encoding="utf-8") as input_handle:
            for line in input_handle:
                document = CanonicalDocument.from_json_line(line)
                _offer_heldout(
                    heldout_heaps.setdefault(document.source_id, []),
                    document=document,
                    line=line,
                    text_bytes=len(document.text.encode("utf-8")),
                    limit=heldout_documents_per_source,
                    seed=seed + 1,
                )
        undersized_sources = sorted(
            source_id
            for source_id, entries in heldout_heaps.items()
            if len(entries) != heldout_documents_per_source
        )
        if undersized_sources:
            raise PublicTokenizerSnapshotError(
                "sources contain fewer held-out rows than required: "
                + ", ".join(undersized_sources)
            )
        heldout_ids = {
            document_id
            for entries in heldout_heaps.values()
            for _negative_score, document_id, _line, _text_count in entries
        }
        with (
            (source / "documents.jsonl").open(encoding="utf-8") as input_handle,
            temporary.open("w", encoding="utf-8", newline="\n") as output_handle,
        ):
            for line in input_handle:
                document = CanonicalDocument.from_json_line(line)
                if document.document_id in heldout_ids or not _selected(
                    document.document_id, sample_ratio, seed
                ):
                    continue
                output_handle.write(line)
                count = len(document.text.encode("utf-8"))
                document_count += 1
                text_bytes += count
                language_bytes[document.language] += count
                content_bytes[document.content_type] += count
                source_bytes[document.source_id] += count
            output_handle.flush()
            os.fsync(output_handle.fileno())
        with heldout_temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as heldout_handle:
            for source_id in sorted(heldout_heaps):
                entries = sorted(heldout_heaps[source_id], key=lambda item: -item[0])
                for _negative_score, _document_id, line, text_count in entries:
                    heldout_handle.write(line)
                    heldout_documents[source_id] += 1
                    heldout_text_bytes[source_id] += text_count
            heldout_handle.flush()
            os.fsync(heldout_handle.fileno())
        if document_count == 0:
            raise PublicTokenizerSnapshotError("snapshot selected no documents")
        os.replace(temporary, documents)
        os.replace(heldout_temporary, heldout)
        documents_sha = _sha256(documents)
        source_name = str(source_manifest["name"])
        data_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        data_version_id = f"data-{source_name}-{ratio_label}-{data_digest[:12]}"
        training_configs = []
        for vocab_size in candidate_vocab_sizes:
            size_label = f"{vocab_size // 1000}k"
            tokenizer_name = f"atom-tokenizer-en-zh-{artifact_label}-{size_label}-v1"
            candidate_output = tokenizer_output / tokenizer_name
            config = _training_config(
                name=tokenizer_name,
                vocab_size=vocab_size,
                data_version_id=data_version_id,
                document_count=document_count,
                documents_sha256=documents_sha,
                documents_path=documents.relative_to(root),
                tokenizer_output_dir=candidate_output.relative_to(root),
            )
            config_path = output / f"tokenizer-training-{size_label}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            training_configs.append(
                {
                    "vocab_size": vocab_size,
                    "name": config_path.name,
                    "sha256": _sha256(config_path),
                    "tokenizer_output_dir": candidate_output.relative_to(
                        root
                    ).as_posix(),
                }
            )
        manifest = {
            **identity,
            "name": f"{source_name}-{ratio_label}",
            "data_version_id": data_version_id,
            "document_count": document_count,
            "text_bytes": text_bytes,
            "language_text_bytes": dict(sorted(language_bytes.items())),
            "content_text_bytes": dict(sorted(content_bytes.items())),
            "source_text_bytes": dict(sorted(source_bytes.items())),
            "documents": {
                "name": documents.name,
                "size_bytes": documents.stat().st_size,
                "sha256": documents_sha,
            },
            "heldout": {
                "name": heldout.name,
                "document_count": sum(heldout_documents.values()),
                "size_bytes": heldout.stat().st_size,
                "sha256": _sha256(heldout),
                "source_documents": dict(sorted(heldout_documents.items())),
                "source_text_bytes": dict(sorted(heldout_text_bytes.items())),
                "disjoint_from_training_by_construction": True,
            },
            "tokenizer_training_configs": training_configs,
            "tokenizer_output_root": tokenizer_output.relative_to(root).as_posix(),
            "synthetic_training_content": False,
            "local_text_conversion": "none",
            "source_audit": {
                "report_sha256": identity["source_audit_sha256"],
                "training_eligible": audit_report["training_eligible"],
            },
        }
        _write_json(manifest_path, manifest)
        completed_path.write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        return manifest
    except BaseException:
        temporary.unlink(missing_ok=True)
        heldout_temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-output-dir", type=Path, required=True)
    parser.add_argument("--sample-ratio", type=float, required=True)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000, 48000])
    parser.add_argument("--heldout-documents-per-source", type=int, default=100)
    parser.add_argument("--artifact-label")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    manifest = build(
        source_dir=args.source_dir,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        tokenizer_output_dir=args.tokenizer_output_dir,
        sample_ratio=args.sample_ratio,
        candidate_vocab_sizes=tuple(args.vocab_sizes),
        heldout_documents_per_source=args.heldout_documents_per_source,
        artifact_label=args.artifact_label,
        seed=args.seed,
        project_root=args.project_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
