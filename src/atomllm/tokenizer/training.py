"""Deterministic Byte-level BPE training and artifact creation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Keep imports safe for callers that only inspect or evaluate tokenizer artifacts.
# The training entry point configures this explicitly from --workers before BPE work.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tokenizers
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from atomllm.data.schema import CanonicalDocument
from atomllm.tokenizer.config import (
    EXPECTED_SPECIAL_TOKENS,
    TokenizerConfig,
    load_tokenizer_config,
)


TOKENIZER_ARTIFACT_SCHEMA_VERSION = 1
PROBE_TEXTS = {
    "zh-Hans": "今天天气很好，适合验证简体中文编码。",
    "en": "Hello, AtomLLM! Numbers: 12345.",
    "zh-Hant": "這是一段用於測試的繁體中文。",
    "ja": "これは日本語の符号化テストです。",
    "code": "def add(a, b):\n    return a + b\n",
    "math": "∫_0^1 x^2 dx = 1/3; E = mc²",
    "whitespace": "alpha  beta\n\tgamma\r\ndelta",
}


class TokenizerTrainingError(RuntimeError):
    """Raised when tokenizer training or artifact verification fails."""


@dataclass(frozen=True, slots=True)
class TrainingInput:
    path: Path
    document_count: int
    sha256: str


def _configure_training_workers(workers: int) -> None:
    """Configure the Rayon pool before the tokenizer performs parallel work."""
    maximum = os.cpu_count() or 1
    if type(workers) is not int or not 1 <= workers <= maximum:
        raise TokenizerTrainingError(
            f"workers must be an integer between 1 and {maximum}"
        )
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if workers == 1 else "true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any], *, pretty: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerTrainingError(
            f"cannot read tokenizer artifact: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise TokenizerTrainingError(
            f"tokenizer artifact is not an object: {path.name}"
        )
    return value


def _resolve_within_root(root: Path, relative_path: Path, field_name: str) -> Path:
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise TokenizerTrainingError(f"{field_name} resolves outside project root")
    return resolved


def verify_training_input(
    config: TokenizerConfig,
    project_root: str | Path,
) -> TrainingInput:
    """Verify identity, line count, and canonical records in one file scan."""
    root = Path(project_root).resolve()
    input_path = _resolve_within_root(
        root,
        config.training_data.input_path,
        "training_data.input_path",
    )
    if not input_path.is_file():
        raise TokenizerTrainingError(
            f"training data not found: {config.training_data.input_path}"
        )
    digest = hashlib.sha256()
    document_count = 0
    validation_error: UnicodeDecodeError | ValueError | None = None
    with input_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if validation_error is None:
                try:
                    CanonicalDocument.from_json_line(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as error:
                    validation_error = error
            document_count += 1
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != config.training_data.expected_sha256:
        raise TokenizerTrainingError("training data SHA-256 does not match config")
    if validation_error is not None:
        raise validation_error
    if document_count != config.training_data.document_count:
        raise TokenizerTrainingError(
            "training data document count does not match config"
        )
    return TrainingInput(
        path=input_path,
        document_count=document_count,
        sha256=actual_sha256,
    )


def _training_texts(training_input: TrainingInput) -> Iterable[str]:
    with training_input.path.open(encoding="utf-8") as handle:
        for line in handle:
            yield CanonicalDocument.from_json_line(line).text


def _build_tokenizer(config: TokenizerConfig) -> tuple[Tokenizer, BpeTrainer]:
    algorithm = config.algorithm
    tokenizer = Tokenizer(
        BPE(
            dropout=None,
            unk_token="<unk>",
            fuse_unk=algorithm.fuse_unk,
            byte_fallback=algorithm.byte_fallback,
            ignore_merges=algorithm.ignore_merges,
        )
    )
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=algorithm.add_prefix_space,
        trim_offsets=algorithm.trim_offsets,
        use_regex=algorithm.use_regex,
    )
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = [
        AddedToken(
            token.token,
            normalized=False,
            special=True,
        )
        for token in config.special_tokens
    ]
    trainer = BpeTrainer(
        vocab_size=algorithm.vocab_size,
        min_frequency=algorithm.min_frequency,
        show_progress=False,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=algorithm.max_token_length,
    )
    return tokenizer, trainer


def _validate_tokenizer(
    tokenizer: Tokenizer,
    config: TokenizerConfig,
) -> dict[str, dict[str, Any]]:
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != config.algorithm.vocab_size:
        raise TokenizerTrainingError(
            f"trained vocabulary has {actual_vocab_size} entries; "
            f"expected {config.algorithm.vocab_size}"
        )
    for token_id, token, _ in EXPECTED_SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) != token_id:
            raise TokenizerTrainingError(f"special token ID mismatch: {token}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded.ids != [token_id]:
            raise TokenizerTrainingError(f"special token is not atomic: {token}")

    unknown_id = tokenizer.token_to_id("<unk>")
    probe_results: dict[str, dict[str, Any]] = {}
    for suite, text in PROBE_TEXTS.items():
        normalized = unicodedata.normalize("NFC", text)
        encoding = tokenizer.encode(normalized, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        unknown_count = encoding.ids.count(unknown_id)
        if decoded != normalized:
            raise TokenizerTrainingError(f"roundtrip probe failed: {suite}")
        if unknown_count != 0:
            raise TokenizerTrainingError(f"unknown token produced by probe: {suite}")
        probe_results[suite] = {
            "character_count": len(normalized),
            "utf8_bytes": len(normalized.encode("utf-8")),
            "token_count": len(encoding.ids),
            "unknown_count": unknown_count,
            "roundtrip": True,
        }
    return probe_results


def _artifact_files(directory: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = directory / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return files


def _validate_existing_artifact(
    output_dir: Path,
    config: TokenizerConfig,
    config_sha256: str,
    training_input: TrainingInput,
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    completed_path = output_dir / "COMPLETED"
    if not output_dir.exists():
        return None
    if (
        not output_dir.is_dir()
        or not manifest_path.is_file()
        or not completed_path.is_file()
    ):
        raise TokenizerTrainingError("existing tokenizer artifact is incomplete")
    manifest = _read_json(manifest_path)
    if manifest.get("config_sha256") != config_sha256:
        raise TokenizerTrainingError(
            "existing tokenizer artifact uses a different config"
        )
    training_data = manifest.get("training_data")
    if not isinstance(training_data, dict):
        raise TokenizerTrainingError(
            "existing tokenizer manifest has invalid training data"
        )
    if (
        training_data.get("sha256") != training_input.sha256
        or training_data.get("document_count") != training_input.document_count
    ):
        raise TokenizerTrainingError(
            "existing tokenizer artifact uses different training data"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TokenizerTrainingError("existing tokenizer manifest has invalid files")
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise TokenizerTrainingError("existing tokenizer file metadata is invalid")
        path = output_dir / name
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            raise TokenizerTrainingError(
                f"tokenizer artifact file SHA-256 mismatch: {name}"
            )
    manifest_sha256 = _sha256(manifest_path)
    if completed_path.read_text(encoding="utf-8") != (
        f"{manifest_sha256}  manifest.json\n"
    ):
        raise TokenizerTrainingError("tokenizer COMPLETED marker is invalid")
    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    _validate_tokenizer(tokenizer, config)
    return manifest


def train_tokenizer(
    config_path: str | Path,
    project_root: str | Path = ".",
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Train or idempotently verify one tokenizer artifact."""
    _configure_training_workers(workers)
    root = Path(project_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = (root / resolved_config_path).resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"tokenizer config not found: {resolved_config_path}")
    config = load_tokenizer_config(resolved_config_path)
    config_sha256 = _sha256(resolved_config_path)
    training_input = verify_training_input(config, root)
    output_dir = _resolve_within_root(root, config.output_dir, "output_dir")
    existing = _validate_existing_artifact(
        output_dir,
        config,
        config_sha256,
        training_input,
    )
    if existing is not None:
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        tokenizer, trainer = _build_tokenizer(config)
        tokenizer.train_from_iterator(
            _training_texts(training_input),
            trainer=trainer,
            length=training_input.document_count,
        )
        probe_results = _validate_tokenizer(tokenizer, config)

        tokenizer_path = temporary_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path), pretty=True)
        model_paths = [
            Path(path) for path in tokenizer.model.save(str(temporary_dir), config.name)
        ]
        config_snapshot_path = temporary_dir / "config.yaml"
        shutil.copyfile(resolved_config_path, config_snapshot_path)

        reloaded = Tokenizer.from_file(str(tokenizer_path))
        reloaded_probe_results = _validate_tokenizer(reloaded, config)
        if reloaded_probe_results != probe_results:
            raise TokenizerTrainingError("reloaded tokenizer probe results changed")

        artifact_names = [
            tokenizer_path.name,
            config_snapshot_path.name,
            *(path.name for path in model_paths),
        ]
        files = _artifact_files(temporary_dir, artifact_names)
        special_tokens = [
            {
                "id": token.token_id,
                "token": token.token,
                "purpose": token.purpose,
            }
            for token in config.special_tokens
        ]
        payload = {
            "schema_version": TOKENIZER_ARTIFACT_SCHEMA_VERSION,
            "name": config.name,
            "status": config.status,
            "training_eligible": config.training_eligible,
            "config_sha256": config_sha256,
            "training_data": {
                "data_version_id": config.training_data.data_version_id,
                "split": config.training_data.split,
                "document_count": training_input.document_count,
                "sha256": training_input.sha256,
            },
            "algorithm": asdict(config.algorithm),
            "model_max_length": config.model_max_length,
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "special_tokens": special_tokens,
            "probe_results": probe_results,
            "library_versions": {
                "python": platform.python_version(),
                "tokenizers": tokenizers.__version__,
                "tokenizers_parallelism": os.environ["TOKENIZERS_PARALLELISM"],
                "tokenizer_training_workers": workers,
            },
            "files": files,
        }
        identity = _canonical_json(payload, pretty=False)
        artifact_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        manifest = {
            **payload,
            "artifact_id": f"tokenizer-{config.name}-{artifact_digest[:12]}",
            "identity_sha256": artifact_digest,
        }
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(
            f"{_canonical_json(manifest, pretty=True)}\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256(manifest_path)
        (temporary_dir / "COMPLETED").write_text(
            f"{manifest_sha256}  manifest.json\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and verify the AtomLLM Byte-level BPE tokenizer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/smoke-32k.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Rayon worker threads used for BPE training (default: 1, low-memory). "
            "Values above 1 can be faster but substantially increase peak memory."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = train_tokenizer(args.config, args.project_root, workers=args.workers)
    print(
        "Tokenizer training complete: "
        f"{manifest['artifact_id']}, "
        f"vocab_size={manifest['vocab_size']}, "
        f"training_eligible={str(manifest['training_eligible']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
