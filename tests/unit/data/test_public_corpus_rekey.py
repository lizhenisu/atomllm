import hashlib
import json
from pathlib import Path

from atomllm.data.public_corpus_rekey import rekey
from atomllm.data.schema import CanonicalDocument, make_document_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rekey_binds_identity_to_content_and_updates_freeze(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    documents = corpus / "documents.jsonl"
    rows = []
    for text in ("first distinct text", "second distinct text"):
        record_id = "reused-upstream-id"
        rows.append(
            CanonicalDocument(
                schema_version=1,
                document_id=make_document_id("public-source", record_id),
                source_id="public-source",
                source_record_id=record_id,
                text=text,
                language="en",
                content_type="general",
                privacy_warnings=(),
                quality_warnings=(),
                metadata={},
            )
        )
    documents.write_text(
        "".join(row.to_json_line() + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "document_count": 2,
        "documents": {
            "name": documents.name,
            "sha256": _sha256(documents),
            "size_bytes": documents.stat().st_size,
        },
    }
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (corpus / "COMPLETED").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
    )

    result = rekey(corpus)

    rekeyed = [
        CanonicalDocument.from_json_line(line)
        for line in documents.read_text(encoding="utf-8").splitlines()
    ]
    assert result["changed_documents"] == 2
    assert len({row.document_id for row in rekeyed}) == 2
    assert all("#sha256-" in row.source_record_id for row in rekeyed)
    assert (corpus / "COMPLETED").read_text(encoding="utf-8") == (
        f"{_sha256(manifest_path)}  manifest.json\n"
    )
    assert rekey(corpus)["changed_documents"] == 0
