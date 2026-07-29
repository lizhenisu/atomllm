import json
from pathlib import Path

import pytest

from atomllm.data.acquisition import (
    AcquisitionError,
    WikipediaAcquisitionRequest,
    acquire_wikipedia_records,
    chinese_script_classifier_identity,
    classify_chinese_script,
)
from atomllm.data.schema import Acquisition, DataSource


def make_source() -> DataSource:
    return DataSource(
        source_id="wikipedia-test-v1",
        name="Synthetic Wikipedia Fixture",
        version="v1",
        license="CC0-1.0",
        homepage="https://example.invalid/wikipedia",
        languages=("zh",),
        content_types=("encyclopedia",),
        data_format="parquet",
        enabled=False,
        acquisition=Acquisition(
            provider="huggingface",
            location="example-invalid/wikipedia",
            revision="0123456789abcdef0123456789abcdef01234567",
            expected_sha256=None,
        ),
    )


def make_records(count: int) -> list[dict[str, str]]:
    return [
        {
            "id": str(index),
            "title": f"合成词条{index}",
            "url": f"https://example.invalid/wiki/{index}",
            "text": f"这是第{index}条合成百科文本。",
        }
        for index in range(count)
    ]


def make_request(limit: int = 3) -> WikipediaAcquisitionRequest:
    return WikipediaAcquisitionRequest(
        source=make_source(),
        config_name="synthetic.zh",
        split="train",
        limit=limit,
    )


def test_chinese_script_classification() -> None:
    assert classify_chinese_script("这是简体中文测试。") == "zh-Hans"
    assert classify_chinese_script("這是繁體中文測試。") == "zh-Hant"
    assert classify_chinese_script("中文") == "zh"


def test_chinese_script_classification_handles_embedded_null() -> None:
    assert classify_chinese_script("这是\x00简体中文测试。") == "zh-Hans"


def test_chinese_script_classifier_has_locked_rule_fingerprint() -> None:
    identity = chinese_script_classifier_identity()

    assert identity["backend"] == "opencc-python-reimplemented"
    assert identity["distribution_version"] == "0.1.7"
    assert len(identity["rules_sha256"]) == 64


def test_acquisition_writes_manifest_and_canonical_documents(
    tmp_path: Path,
) -> None:
    manifest = acquire_wikipedia_records(make_records(4), make_request(), tmp_path)

    assert manifest["record_count"] == 3
    assert manifest["source_enabled"] is False
    assert manifest["language_counts"] == {"zh-Hans": 3}
    assert len(manifest["documents_sha256"]) == 64
    lines = (tmp_path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["source_record_id"] == "0"
    assert json.loads(lines[0])["metadata"]["url"].startswith(
        "https://example.invalid/"
    )


def test_acquisition_resumes_after_iterator_failure(tmp_path: Path) -> None:
    records = make_records(3)

    def interrupted_records():
        yield records[0]
        yield records[1]
        raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        acquire_wikipedia_records(interrupted_records(), make_request(), tmp_path)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["records_written"] == 2
    assert state["completed"] is False

    manifest = acquire_wikipedia_records(records, make_request(), tmp_path)

    assert manifest["record_count"] == 3
    lines = (tmp_path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["source_record_id"] for line in lines] == ["0", "1", "2"]


def test_completed_acquisition_is_idempotent(tmp_path: Path) -> None:
    first = acquire_wikipedia_records(make_records(3), make_request(), tmp_path)
    second = acquire_wikipedia_records([], make_request(), tmp_path)

    assert second == first


def test_resume_rejects_mismatched_request(tmp_path: Path) -> None:
    acquire_wikipedia_records(make_records(3), make_request(), tmp_path)

    with pytest.raises(AcquisitionError, match="does not match request field: limit"):
        acquire_wikipedia_records(make_records(4), make_request(limit=4), tmp_path)
