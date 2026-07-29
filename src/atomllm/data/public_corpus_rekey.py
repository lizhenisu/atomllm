"""Re-key a frozen public tokenizer corpus with content-bound record IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from atomllm.data.public_tokenizer_corpus import _content_bound_source_record_id
from atomllm.data.schema import CanonicalDocument, make_document_id


class PublicCorpusRekeyError(RuntimeError):
    """Raised when a frozen corpus cannot be safely re-keyed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def rekey(corpus_dir: Path) -> dict[str, Any]:
    corpus = corpus_dir.resolve()
    documents = corpus / "documents.jsonl"
    manifest_path = corpus / "manifest.json"
    completed = corpus / "COMPLETED"
    state_path = corpus / "state.json"
    if not all(path.is_file() for path in (documents, manifest_path, completed)):
        raise PublicCorpusRekeyError("corpus must be frozen before re-keying")
    manifest_sha = _sha256(manifest_path)
    if completed.read_text(encoding="utf-8") != f"{manifest_sha}  manifest.json\n":
        raise PublicCorpusRekeyError("COMPLETED does not match manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["documents"]
    if (
        documents.stat().st_size != expected["size_bytes"]
        or _sha256(documents) != expected["sha256"]
    ):
        raise PublicCorpusRekeyError("documents do not match manifest")

    temporary = corpus / "documents.rekeying.jsonl"
    count = 0
    changed = 0
    with (
        documents.open(encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            document = CanonicalDocument.from_json_line(line)
            digest = hashlib.sha256(document.text.encode("utf-8")).digest()
            record_id = _content_bound_source_record_id(
                document.source_record_id, digest
            )
            if record_id != document.source_record_id:
                changed += 1
                mapping = document.to_mapping()
                mapping["source_record_id"] = record_id
                mapping["document_id"] = make_document_id(document.source_id, record_id)
                document = CanonicalDocument.from_mapping(mapping)
            destination.write(document.to_json_line() + "\n")
            count += 1
            if count % 100_000 == 0:
                print(f"[public-corpus-rekey] documents={count}", flush=True)
        destination.flush()
        os.fsync(destination.fileno())
    if count != manifest["document_count"]:
        temporary.unlink(missing_ok=True)
        raise PublicCorpusRekeyError("document count does not match manifest")

    new_sha = _sha256(temporary)
    new_size = temporary.stat().st_size
    temporary.replace(documents)
    manifest["documents"] = {
        "name": documents.name,
        "sha256": new_sha,
        "size_bytes": new_size,
    }
    manifest["document_identity"] = {
        "source_record_id": "upstream-id#sha256-<normalized-text-sha256>",
        "purpose": "disambiguate upstream IDs reused for different content",
    }
    _write_json(manifest_path, manifest)
    completed.write_text(f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8")
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["committed_output_bytes"] = new_size
        _write_json(state_path, state)
    fingerprints = corpus / "fingerprints.sqlite3"
    if fingerprints.is_file():
        with sqlite3.connect(fingerprints) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                ("committed_output_bytes", str(new_size)),
            )
            connection.commit()
    result = {
        "document_count": count,
        "changed_documents": changed,
        "documents_sha256": new_sha,
        "documents_size_bytes": new_size,
        "manifest_sha256": _sha256(manifest_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    args = parser.parse_args(argv)
    rekey(args.corpus_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
