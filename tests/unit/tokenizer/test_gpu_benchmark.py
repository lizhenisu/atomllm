from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.tokenizer.gpu_benchmark import (
    _candidate_model,
    _packed_public_tokens,
)


def test_candidate_model_accounts_for_selected_vocabulary() -> None:
    candidate = _candidate_model(
        Path("configs/model/atom-base-300m.yaml"),
        {"vocab_size": 48000},
        "a" * 64,
    )

    assert candidate.tokenizer.vocab_size == 48000
    assert candidate.expected_parameter_count == 319_734_784


def test_packed_public_tokens_use_real_documents_and_boundaries(tmp_path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "alpha": 4},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    source_id = "public-test"
    record_id = "record-1"
    document = CanonicalDocument(
        schema_version=1,
        document_id=make_document_id(source_id, record_id),
        source_id=source_id,
        source_record_id=record_id,
        text="alpha",
        language="en",
        content_type="general",
        privacy_warnings=(),
        quality_warnings=(),
        metadata={},
    )
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(document.to_json_line() + "\n", encoding="utf-8")

    tokens = _packed_public_tokens(heldout, tokenizer, required_tokens=6)

    assert tokens.tolist() == [2, 4, 3, 2, 4, 3]
