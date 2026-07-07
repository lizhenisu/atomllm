from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from atomllm.data.schema import CanonicalDocument, make_document_id
from atomllm.training.config import file_sha256
from atomllm.training.packing import PackingSpec, pack_token_dataset


@pytest.fixture
def packed_dataset_dir(tmp_path: Path) -> Path:
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        **{f"doc{index}": index + 4 for index in range(10)},
        "tail": 14,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    input_path = tmp_path / "train.jsonl"
    lines = []
    for index in range(10):
        source_record_id = f"synthetic-{index}"
        document = CanonicalDocument(
            schema_version=1,
            document_id=make_document_id("synthetic", source_record_id),
            source_id="synthetic",
            source_record_id=source_record_id,
            text=f"doc{index} tail",
            language="en",
            content_type="general",
            privacy_warnings=(),
            quality_warnings=(),
            metadata={},
        )
        lines.append(document.to_json_line())
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_dir = tmp_path / "packed"
    pack_token_dataset(
        PackingSpec(
            name="synthetic-seq4",
            data_version_id="synthetic-data-v1",
            input_path=input_path,
            input_sha256=file_sha256(input_path),
            tokenizer_version_id="synthetic-tokenizer-v1",
            tokenizer_path=tokenizer_path,
            tokenizer_sha256=file_sha256(tokenizer_path),
            vocab_size=15,
            sequence_length=4,
            output_dir=output_dir,
        )
    )
    return output_dir
