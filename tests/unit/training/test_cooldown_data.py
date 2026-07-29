from __future__ import annotations

from pathlib import Path

from atomllm.data.schema import CanonicalDocument
from atomllm.training.cooldown_data import _load_config, _selected


def _document(**changes: object) -> CanonicalDocument:
    values = {
        "schema_version": 1,
        "document_id": "doc-" + "0" * 64,
        "source_id": "wikipedia-20231101",
        "source_record_id": "record",
        "text": "useful text",
        "language": "zh-Hans",
        "content_type": "encyclopedia",
        "privacy_warnings": (),
        "quality_warnings": (),
        "metadata": {"estimated_tokens": 10},
    }
    values.update(changes)
    return CanonicalDocument(**values)


def test_cooldown_config_freezes_warning_rejection_and_balanced_rules() -> None:
    config = _load_config(
        Path("configs/training/atom-base-300m-cooldown-data-pilot-v1.yaml")
    )

    assert config["require_no_quality_warnings"] is True
    assert config["require_no_privacy_warnings"] is True
    assert {rule["name"] for rule in config["rules"]} == {
        "encyclopedia",
        "science",
        "math",
        "code",
        "chinese-general",
        "english-general",
    }


def test_cooldown_selection_rejects_all_warning_types() -> None:
    config = _load_config(
        Path("configs/training/atom-base-300m-cooldown-data-pilot-v1.yaml")
    )

    assert _selected(_document(quality_warnings=("high_repetition",)), config) is None
    assert _selected(_document(privacy_warnings=("email",)), config) is None


def test_cooldown_selection_is_deterministic() -> None:
    config = _load_config(
        Path("configs/training/atom-base-300m-cooldown-data-pilot-v1.yaml")
    )
    document = _document()

    assert _selected(document, config) == _selected(document, config)
