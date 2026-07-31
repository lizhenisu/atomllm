from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atomllm.tokenizer.technical_evaluation import (
    DocumentMetric,
    ExternalProbe,
    _aggregate,
    _comparison,
    _document_distribution,
    _load_external_probe,
)


def _rows(token_count: int) -> list[DocumentMetric]:
    return [
        DocumentMetric("source-a", "en", "general", 100, 100, token_count, 0, 0),
        DocumentMetric("source-a", "en", "general", 50, 60, 25, 0, 0),
    ]


def test_aggregate_reports_weighted_compression() -> None:
    result = _aggregate(_rows(25))

    assert result["document_count"] == 2
    assert result["character_count"] == 150
    assert result["token_count"] == 50
    assert result["characters_per_token"] == 3.0
    assert result["bytes_per_token"] == 3.2


def test_document_distribution_reports_requested_quantiles() -> None:
    result = _document_distribution(_rows(50))

    assert result["p50_characters_per_token"] == 2.0
    assert result["p05_characters_per_token"] == 2.0
    assert result["p95_characters_per_token"] == 2.0


def test_comparison_reports_candidate_token_increase() -> None:
    candidate = {
        "summary": {"token_count": 110},
        "by_language": {"en": {"token_count": 110}},
        "by_content_type": {"general": {"token_count": 110}},
    }
    comparison = {
        "summary": {"token_count": 100},
        "by_language": {"en": {"token_count": 100}},
        "by_content_type": {"general": {"token_count": 100}},
    }

    result = _comparison(candidate, comparison)

    assert result["overall"]["candidate_token_increase_percent"] == 10.0
    assert result["by_language"]["en"]["candidate_tokens"] == 110


def test_external_probe_sampling_is_deterministic(tmp_path: Path) -> None:
    parquet_path = tmp_path / "probe.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b", "c", "d"],
                "text": ["A" * 200, "B" * 210, "short", "D" * 220],
            }
        ),
        parquet_path,
    )
    probe = ExternalProbe(
        source_id="external-en",
        parquet_file=Path("probe.parquet"),
        text_field="text",
        id_field="id",
        language="en",
        content_type="general",
        sample_size=2,
        minimum_characters=200,
        maximum_characters=500,
    )

    first, metadata = _load_external_probe(tmp_path, probe, seed=42)
    second, _ = _load_external_probe(tmp_path, probe, seed=42)

    assert first == second
    assert len(first) == 2
    assert metadata["eligible_rows"] == 3
    assert metadata["selection_method"] == "lowest-sha256-of-seed-and-record-id"
