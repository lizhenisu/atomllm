from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context

import pytest

from atomllm.data.public_corpus_audit import (
    PublicCorpusAuditError,
    _audit_document,
    _audit_document_worker,
    _initialize_audit_worker,
)
from atomllm.data.public_tokenizer_corpus import load_config
from atomllm.data.schema import SCHEMA_VERSION, CanonicalDocument, make_document_id


def _document(*, text: str, source_id: str, score: float) -> CanonicalDocument:
    return CanonicalDocument.from_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "document_id": make_document_id(source_id, "row-1"),
            "source_id": source_id,
            "source_record_id": "row-1",
            "text": text,
            "language": "zh-Hans",
            "content_type": "general",
            "privacy_warnings": [],
            "quality_warnings": [],
            "metadata": {
                "local_text_conversion": "none",
                "upstream_quality_score": score,
                "upstream_quality_score_field": "score",
            },
        }
    )


def test_audit_document_accepts_quality_scored_simplified_text() -> None:
    source = load_config().sources[0]
    text = "高质量公开语料必须具有稳定来源、清晰内容和可审计的质量评分。" * 20
    document = _document(text=text, source_id=source.source_id, score=4.2)

    size, digest = _audit_document(document, source)

    assert size == len(text.encode("utf-8"))
    assert len(digest) == 32


def test_audit_document_rejects_score_below_source_threshold() -> None:
    source = load_config().sources[0]
    text = "这段文本长度足够，但上游质量评分没有达到正式数据门槛。" * 20
    document = _document(text=text, source_id=source.source_id, score=3.99)

    with pytest.raises(PublicCorpusAuditError, match="quality threshold"):
        _audit_document(document, source)


def test_audit_document_rejects_local_conversion_marker() -> None:
    source = load_config().sources[0]
    text = "正式语料不允许在本地执行繁体中文到简体中文的文本转换。" * 20
    document = _document(text=text, source_id=source.source_id, score=4.5)
    document = replace(
        document, metadata={**document.metadata, "local_text_conversion": "t2s"}
    )

    with pytest.raises(PublicCorpusAuditError, match="local text conversion"):
        _audit_document(document, source)


def test_parallel_audit_matches_locked_single_process_result() -> None:
    source = load_config().sources[0]
    texts = [
        "高质量公开语料必须具有稳定来源和可审计的质量评分。" * 20,
        "可靠的并行审计不能改变中文脚本分类或质量判断结果。" * 20,
    ]
    documents = [
        _document(text=text, source_id=source.source_id, score=4.5) for text in texts
    ]

    expected = [_audit_document(document, source) for document in documents]
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
        initializer=_initialize_audit_worker,
        initargs=({source.source_id: source},),
    ) as executor:
        actual = list(executor.map(_audit_document_worker, documents))

    assert actual == expected
