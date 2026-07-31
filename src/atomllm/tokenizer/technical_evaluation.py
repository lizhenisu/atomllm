"""Produce a reproducible technical report for a release tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import yaml
from tokenizers import Tokenizer

from atomllm.data.schema import CanonicalDocument
from atomllm.tokenizer.evaluation import verify_tokenizer_directory


SCHEMA_VERSION = 1
_WORKER_TOKENIZER: Tokenizer | None = None


class TechnicalEvaluationError(RuntimeError):
    """Raised when a technical tokenizer evaluation is not trustworthy."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    name: str
    candidate_tokenizer: Path
    comparison_tokenizer: Path
    snapshot_dir: Path
    pretraining_plan: Path
    output_dir: Path
    report_path: Path
    quality_workers: int
    bootstrap_resamples: int
    bootstrap_seed: int
    external_probes: tuple[ExternalProbe, ...]
    robustness_suites: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    source_id: str
    language: str
    content_type: str
    text: str


@dataclass(frozen=True, slots=True)
class ExternalProbe:
    source_id: str
    parquet_file: Path
    text_field: str
    id_field: str
    language: str
    content_type: str
    sample_size: int
    minimum_characters: int
    maximum_characters: int


@dataclass(frozen=True, slots=True)
class DocumentMetric:
    source_id: str
    language: str
    content_type: str
    characters: int
    utf8_bytes: int
    tokens: int
    unknowns: int
    roundtrip_failure: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TechnicalEvaluationError(f"{field} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise TechnicalEvaluationError(f"{field} must be a safe relative path")
    return path


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise TechnicalEvaluationError(f"{field} must be a positive integer")
    return value


def load_config(path: Path) -> EvaluationConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TechnicalEvaluationError(f"cannot read config: {path}") from error
    if not isinstance(raw, dict):
        raise TechnicalEvaluationError("evaluation config must be a mapping")
    expected = {
        "schema_version",
        "name",
        "candidate_tokenizer",
        "comparison_tokenizer",
        "snapshot_dir",
        "pretraining_plan",
        "output_dir",
        "report_path",
        "quality_workers",
        "bootstrap_resamples",
        "bootstrap_seed",
        "external_probes",
        "robustness_suites",
    }
    if set(raw) != expected:
        raise TechnicalEvaluationError(
            f"evaluation config keys differ: {sorted(set(raw) ^ expected)}"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise TechnicalEvaluationError(f"schema_version must be {SCHEMA_VERSION}")
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise TechnicalEvaluationError("name must be a non-empty string")
    suites = raw["robustness_suites"]
    if not isinstance(suites, dict) or not suites:
        raise TechnicalEvaluationError("robustness_suites must be a mapping")
    parsed_suites: dict[str, tuple[str, ...]] = {}
    for suite, samples in suites.items():
        if (
            not isinstance(suite, str)
            or not suite
            or not isinstance(samples, list)
            or not samples
            or not all(isinstance(sample, str) and sample for sample in samples)
        ):
            raise TechnicalEvaluationError("robustness suites are invalid")
        parsed_suites[suite] = tuple(samples)
    seed = raw["bootstrap_seed"]
    if type(seed) is not int or seed < 0:
        raise TechnicalEvaluationError("bootstrap_seed must be non-negative")
    raw_probes = raw["external_probes"]
    if not isinstance(raw_probes, list) or not raw_probes:
        raise TechnicalEvaluationError("external_probes must be a non-empty list")
    probe_keys = {
        "source_id",
        "parquet_file",
        "text_field",
        "id_field",
        "language",
        "content_type",
        "sample_size",
        "minimum_characters",
        "maximum_characters",
    }
    probes: list[ExternalProbe] = []
    for probe in raw_probes:
        if not isinstance(probe, dict) or set(probe) != probe_keys:
            raise TechnicalEvaluationError("external probe keys are invalid")
        string_fields = (
            "source_id",
            "text_field",
            "id_field",
            "language",
            "content_type",
        )
        if any(
            not isinstance(probe[field], str) or not probe[field]
            for field in string_fields
        ):
            raise TechnicalEvaluationError("external probe strings are invalid")
        minimum = _positive_int(
            probe["minimum_characters"], "external probe minimum_characters"
        )
        maximum = _positive_int(
            probe["maximum_characters"], "external probe maximum_characters"
        )
        if minimum > maximum:
            raise TechnicalEvaluationError("external probe character range is invalid")
        probes.append(
            ExternalProbe(
                source_id=probe["source_id"],
                parquet_file=_safe_path(
                    probe["parquet_file"], "external probe parquet_file"
                ),
                text_field=probe["text_field"],
                id_field=probe["id_field"],
                language=probe["language"],
                content_type=probe["content_type"],
                sample_size=_positive_int(
                    probe["sample_size"], "external probe sample_size"
                ),
                minimum_characters=minimum,
                maximum_characters=maximum,
            )
        )
    return EvaluationConfig(
        name=name,
        candidate_tokenizer=_safe_path(
            raw["candidate_tokenizer"], "candidate_tokenizer"
        ),
        comparison_tokenizer=_safe_path(
            raw["comparison_tokenizer"], "comparison_tokenizer"
        ),
        snapshot_dir=_safe_path(raw["snapshot_dir"], "snapshot_dir"),
        pretraining_plan=_safe_path(raw["pretraining_plan"], "pretraining_plan"),
        output_dir=_safe_path(raw["output_dir"], "output_dir"),
        report_path=_safe_path(raw["report_path"], "report_path"),
        quality_workers=_positive_int(raw["quality_workers"], "quality_workers"),
        bootstrap_resamples=_positive_int(
            raw["bootstrap_resamples"], "bootstrap_resamples"
        ),
        bootstrap_seed=seed,
        external_probes=tuple(probes),
        robustness_suites=parsed_suites,
    )


def _resolve(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise TechnicalEvaluationError(f"path resolves outside project: {path}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TechnicalEvaluationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise TechnicalEvaluationError(f"JSON must contain an object: {path}")
    return value


def _load_documents(
    snapshot_dir: Path,
) -> tuple[list[EvaluationDocument], dict[str, Any], Path]:
    manifest_path = snapshot_dir / "manifest.json"
    completed_path = snapshot_dir / "COMPLETED"
    heldout_path = snapshot_dir / "heldout.jsonl"
    if not all(
        path.is_file() for path in (manifest_path, completed_path, heldout_path)
    ):
        raise TechnicalEvaluationError("snapshot is incomplete")
    if completed_path.read_text(encoding="utf-8") != (
        f"{_sha256(manifest_path)}  manifest.json\n"
    ):
        raise TechnicalEvaluationError("snapshot completion marker is invalid")
    manifest = _read_json(manifest_path)
    heldout = manifest.get("heldout")
    if not isinstance(heldout, dict):
        raise TechnicalEvaluationError("snapshot heldout metadata is invalid")
    if heldout.get("disjoint_from_training_by_construction") is not True:
        raise TechnicalEvaluationError("heldout is not declared training-disjoint")
    if heldout.get("sha256") != _sha256(heldout_path):
        raise TechnicalEvaluationError("heldout SHA-256 mismatch")
    documents: list[EvaluationDocument] = []
    source_counts: Counter[str] = Counter()
    with heldout_path.open(encoding="utf-8") as handle:
        for line in handle:
            document = CanonicalDocument.from_json_line(line)
            text = unicodedata.normalize("NFC", document.text)
            documents.append(
                EvaluationDocument(
                    source_id=document.source_id,
                    language=document.language,
                    content_type=document.content_type,
                    text=text,
                )
            )
            source_counts[document.source_id] += 1
    if len(documents) != heldout.get("document_count"):
        raise TechnicalEvaluationError("heldout document count mismatch")
    if dict(sorted(source_counts.items())) != heldout.get("source_documents"):
        raise TechnicalEvaluationError("heldout source counts mismatch")
    return documents, manifest, heldout_path


def _verify_lineage(
    tokenizer_manifest: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    documents = snapshot.get("documents")
    if not isinstance(documents, dict):
        raise TechnicalEvaluationError("snapshot documents metadata is invalid")
    expected = {
        "data_version_id": snapshot.get("data_version_id"),
        "split": "train",
        "document_count": snapshot.get("document_count"),
        "sha256": documents.get("sha256"),
    }
    if tokenizer_manifest.get("training_data") != expected:
        raise TechnicalEvaluationError("tokenizer and snapshot lineage differ")


def _load_external_probe(
    root: Path, probe: ExternalProbe, seed: int
) -> tuple[list[EvaluationDocument], dict[str, Any]]:
    parquet_path = _resolve(root, probe.parquet_file)
    if not parquet_path.is_file():
        raise TechnicalEvaluationError(
            f"external probe parquet file is missing: {probe.parquet_file}"
        )
    parquet = pq.ParquetFile(parquet_path)
    fields = set(parquet.schema.names)
    if probe.text_field not in fields or probe.id_field not in fields:
        raise TechnicalEvaluationError("external probe parquet fields are missing")
    table = parquet.read(columns=[probe.id_field, probe.text_field])
    record_ids = table.column(probe.id_field).to_pylist()
    texts = table.column(probe.text_field).to_pylist()
    scored: list[tuple[bytes, str]] = []
    eligible_count = 0
    for record_id, raw_text in zip(record_ids, texts, strict=True):
        if not isinstance(raw_text, str):
            continue
        text = unicodedata.normalize("NFC", raw_text)
        if not probe.minimum_characters <= len(text) <= probe.maximum_characters:
            continue
        eligible_count += 1
        identity = f"{seed}\0{record_id}".encode("utf-8", errors="surrogatepass")
        scored.append((hashlib.sha256(identity).digest(), text))
    if len(scored) < probe.sample_size:
        raise TechnicalEvaluationError(
            f"external probe has only {len(scored)} eligible rows"
        )
    selected = sorted(scored, key=lambda item: item[0])[: probe.sample_size]
    documents = [
        EvaluationDocument(
            source_id=probe.source_id,
            language=probe.language,
            content_type=probe.content_type,
            text=text,
        )
        for _, text in selected
    ]
    return documents, {
        "source_id": probe.source_id,
        "parquet_file": probe.parquet_file.as_posix(),
        "parquet_sha256": _sha256(parquet_path),
        "parquet_rows": parquet.metadata.num_rows,
        "eligible_rows": eligible_count,
        "sample_size": probe.sample_size,
        "selection_method": "lowest-sha256-of-seed-and-record-id",
        "minimum_characters": probe.minimum_characters,
        "maximum_characters": probe.maximum_characters,
        "tokenizer_training_disjoint_by_source": True,
    }


def _worker_init(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)


def _quality_chunk(
    documents: Sequence[EvaluationDocument],
) -> tuple[list[DocumentMetric], dict[int, int]]:
    if _WORKER_TOKENIZER is None:
        raise TechnicalEvaluationError("worker tokenizer is not initialized")
    tokenizer = _WORKER_TOKENIZER
    unknown_id = tokenizer.token_to_id("<unk>")
    if unknown_id is None:
        raise TechnicalEvaluationError("tokenizer has no <unk> token")
    texts = [document.text for document in documents]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    ids = [encoding.ids for encoding in encodings]
    decoded = tokenizer.decode_batch(ids, skip_special_tokens=False)
    frequencies: Counter[int] = Counter()
    rows: list[DocumentMetric] = []
    for document, token_ids, restored in zip(documents, ids, decoded, strict=True):
        frequencies.update(token_ids)
        rows.append(
            DocumentMetric(
                source_id=document.source_id,
                language=document.language,
                content_type=document.content_type,
                characters=len(document.text),
                utf8_bytes=len(document.text.encode("utf-8")),
                tokens=len(token_ids),
                unknowns=token_ids.count(unknown_id),
                roundtrip_failure=int(restored != document.text),
            )
        )
    return rows, dict(frequencies)


def _balanced_chunks(
    documents: Sequence[EvaluationDocument], chunk_count: int
) -> list[list[EvaluationDocument]]:
    bins: list[list[EvaluationDocument]] = [[] for _ in range(chunk_count)]
    sizes = [0] * chunk_count
    for document in sorted(documents, key=lambda item: len(item.text), reverse=True):
        index = min(range(chunk_count), key=sizes.__getitem__)
        bins[index].append(document)
        sizes[index] += len(document.text)
    return [items for items in bins if items]


def _evaluate_parallel(
    tokenizer_path: Path,
    documents: Sequence[EvaluationDocument],
    workers: int,
) -> tuple[list[DocumentMetric], Counter[int]]:
    chunks = _balanced_chunks(documents, min(len(documents), workers * 4))
    rows: list[DocumentMetric] = []
    frequencies: Counter[int] = Counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(tokenizer_path),),
    ) as pool:
        for chunk_rows, chunk_frequencies in pool.map(_quality_chunk, chunks):
            rows.extend(chunk_rows)
            frequencies.update(chunk_frequencies)
    return rows, frequencies


def _aggregate(rows: Sequence[DocumentMetric]) -> dict[str, int | float]:
    documents = len(rows)
    characters = sum(row.characters for row in rows)
    utf8_bytes = sum(row.utf8_bytes for row in rows)
    tokens = sum(row.tokens for row in rows)
    unknowns = sum(row.unknowns for row in rows)
    failures = sum(row.roundtrip_failure for row in rows)
    if not rows or tokens <= 0:
        raise TechnicalEvaluationError("metric group is empty")
    return {
        "document_count": documents,
        "character_count": characters,
        "utf8_bytes": utf8_bytes,
        "token_count": tokens,
        "characters_per_token": round(characters / tokens, 6),
        "bytes_per_token": round(utf8_bytes / tokens, 6),
        "unknown_count": unknowns,
        "unknown_rate": round(unknowns / tokens, 10),
        "roundtrip_failures": failures,
    }


def _group_metrics(
    rows: Sequence[DocumentMetric], attribute: str
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[DocumentMetric]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, attribute))].append(row)
    return {key: _aggregate(value) for key, value in sorted(groups.items())}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise TechnicalEvaluationError("cannot calculate an empty percentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _document_distribution(rows: Sequence[DocumentMetric]) -> dict[str, float]:
    values = [row.characters / row.tokens for row in rows]
    return {
        "p05_characters_per_token": round(_percentile(values, 0.05), 6),
        "p50_characters_per_token": round(_percentile(values, 0.50), 6),
        "p95_characters_per_token": round(_percentile(values, 0.95), 6),
    }


def _bootstrap_interval(
    rows: Sequence[DocumentMetric], resamples: int, seed: int
) -> dict[str, float]:
    groups: dict[str, list[DocumentMetric]] = defaultdict(list)
    for row in rows:
        groups[row.source_id].append(row)
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        characters = 0
        tokens = 0
        for source_rows in groups.values():
            for _ in range(len(source_rows)):
                sampled = generator.choice(source_rows)
                characters += sampled.characters
                tokens += sampled.tokens
        estimates.append(characters / tokens)
    return {
        "characters_per_token_ci95_low": round(_percentile(estimates, 0.025), 6),
        "characters_per_token_ci95_high": round(_percentile(estimates, 0.975), 6),
    }


def _quality_report(
    rows: Sequence[DocumentMetric],
    frequencies: Counter[int],
    vocab_size: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    languages: dict[str, list[DocumentMetric]] = defaultdict(list)
    for row in rows:
        languages[row.language].append(row)
    by_language = _group_metrics(rows, "language")
    for index, (language, language_rows) in enumerate(sorted(languages.items())):
        by_language[language].update(
            _bootstrap_interval(language_rows, resamples, seed + index + 1)
        )
    summary = _aggregate(rows)
    summary.update(_bootstrap_interval(rows, resamples, seed))
    summary.update(_document_distribution(rows))
    observed = len(frequencies)
    return {
        "summary": summary,
        "by_language": by_language,
        "by_content_type": _group_metrics(rows, "content_type"),
        "by_source": _group_metrics(rows, "source_id"),
        "vocabulary_observation": {
            "observed_token_ids": observed,
            "vocab_size": vocab_size,
            "heldout_vocabulary_coverage": round(observed / vocab_size, 6),
        },
    }


def _comparison(
    candidate: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    def compare_groups(
        left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, float | int]]:
        if set(left) != set(right):
            raise TechnicalEvaluationError("comparison metric groups differ")
        return {
            key: {
                "candidate_tokens": left[key]["token_count"],
                "comparison_tokens": right[key]["token_count"],
                "candidate_token_increase_percent": round(
                    (left[key]["token_count"] / right[key]["token_count"] - 1) * 100,
                    4,
                ),
            }
            for key in sorted(left)
        }

    left_summary = candidate["summary"]
    right_summary = comparison["summary"]
    return {
        "overall": {
            "candidate_tokens": left_summary["token_count"],
            "comparison_tokens": right_summary["token_count"],
            "candidate_token_increase_percent": round(
                (left_summary["token_count"] / right_summary["token_count"] - 1) * 100,
                4,
            ),
        },
        "by_language": compare_groups(
            candidate["by_language"], comparison["by_language"]
        ),
        "by_content_type": compare_groups(
            candidate["by_content_type"], comparison["by_content_type"]
        ),
    }


def _robustness(
    tokenizer: Tokenizer, suites: dict[str, tuple[str, ...]]
) -> dict[str, Any]:
    unknown_id = tokenizer.token_to_id("<unk>")
    if unknown_id is None:
        raise TechnicalEvaluationError("tokenizer has no <unk> token")
    report: dict[str, Any] = {}
    for name, raw_samples in suites.items():
        samples = [unicodedata.normalize("NFC", sample) for sample in raw_samples]
        encodings = tokenizer.encode_batch(samples, add_special_tokens=False)
        ids = [encoding.ids for encoding in encodings]
        decoded = tokenizer.decode_batch(ids, skip_special_tokens=False)
        token_count = sum(len(value) for value in ids)
        report[name] = {
            "sample_count": len(samples),
            "token_count": token_count,
            "unknown_count": sum(value.count(unknown_id) for value in ids),
            "roundtrip_failures": sum(
                expected != actual
                for expected, actual in zip(samples, decoded, strict=True)
            ),
        }
    return report


def _source_label(source_id: str) -> str:
    labels = {
        "cci3-hq-native-zh-hans": "[CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ)",
        "fineweb-edu-en-score4": "[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)",
        "industry-corpus2-zh-high": "[IndustryCorpus2](https://huggingface.co/datasets/BAAI/IndustryCorpus2)",
        "open-web-math-en": "[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math)",
        "pes2o-v3-en-science": "[peS2o v3](https://huggingface.co/datasets/allenai/peS2o)",
        "supplement-dclm-baseline-en": "[DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet)",
        "wikipedia-20231101-en": "[Wikipedia (en)](https://huggingface.co/datasets/wikimedia/wikipedia)",
        "wikipedia-20231101-zh-hans-native": "[Wikipedia (zh)](https://huggingface.co/datasets/wikimedia/wikipedia)",
    }
    if source_id.startswith("starcoderdata-"):
        language = source_id.removeprefix("starcoderdata-")
        return (
            "[StarCoderData]"
            f"(https://huggingface.co/datasets/bigcode/starcoderdata) ({language})"
        )
    return labels.get(source_id, source_id)


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["quality"]["candidate"]
    summary = candidate["summary"]
    compare = report["comparison"]
    lines = [
        "# Atom Tokenizer 32K 技术评测报告",
        "",
        f"> 评测版本：`{report['evaluation_id']}`  ",
        f"> 生成时间：{report['generated_at']}  ",
        f"> Tokenizer SHA-256：`{report['candidate']['tokenizer_sha256']}`",
        "",
        "## 摘要",
        "",
        (
            f"本报告在 {summary['document_count']:,} 篇真实文档组成的合并评测集上"
            f"评测 Atom 32K Byte-level BPE。样本覆盖最终 100B 配方的 20 个公开来源，"
            f"共 {summary['character_count']:,} 字符、{summary['utf8_bytes']:,} UTF-8 "
            "字节。"
        ),
        "",
        (
            f"32K Tokenizer 的总体压缩率为 **{summary['characters_per_token']:.4f} "
            f"字符/Token**（分层 Bootstrap 95% CI "
            f"{summary['characters_per_token_ci95_low']:.4f}–"
            f"{summary['characters_per_token_ci95_high']:.4f}），未知 Token 数和 NFC "
            f"往返失败数均为 **0**。相对 48K 候选，32K 在同一测试集上增加 "
            f"**{compare['overall']['candidate_token_increase_percent']:.2f}%** Token，"
            "但词表与模型参数更少，且此前端到端 GPU 门禁选择了 32K 版本。"
        ),
        "",
        "## 1. 评测对象",
        "",
        "| 项目 | 32K 正式版本 | 48K 对照版本 |",
        "| --- | --- | --- |",
        (
            f"| 词表规模 | {report['candidate']['vocab_size']:,} | "
            f"{report['comparison_tokenizer']['vocab_size']:,} |"
        ),
        (
            f"| Tokenizer 文件 | {report['candidate']['tokenizer_bytes'] / 1024**2:.2f} MiB | "
            f"{report['comparison_tokenizer']['tokenizer_bytes'] / 1024**2:.2f} MiB |"
        ),
        f"| 算法 | {report['candidate']['algorithm']} | {report['comparison_tokenizer']['algorithm']} |",
        "| Unicode 规范化 | NFC | NFC |",
        "| 未知 Token | `<unk>` | `<unk>` |",
        "",
        "## 2. 数据与方法",
        "",
        (
            "质量评测使用 Tokenizer 数据快照中按每个完整来源最低 SHA-256 分数固定"
            "保留的 100 篇文档；这些文档在构造时即从训练集排除。19 个来源包括"
            "简体中文通用文本、中英文 Wikipedia、英文教育网页、数学、科学论文及"
            "12 种代码子集。核心指标为字符/Token、UTF-8 字节/Token、未知 Token "
            "率和 NFC 规范化后的 encode→decode 往返一致性。"
        ),
        "",
        (
            "最终 100B 预训练配方后续加入 DCLM Baseline 补足英文预算。该来源未参与"
            "本 Tokenizer 训练，因此从固定 Parquet 分片按记录 ID 最低 SHA-256 "
            "抽取 1,000 篇，并与原有 19 来源 held-out 合并计算总体、语言、内容类型、"
            "来源及 32K/48K 对照指标。"
        ),
        "",
        (
            f"置信区间使用按来源分层的 {report['methodology']['bootstrap_resamples']:,} "
            "次非参数 Bootstrap，避免长文档或单一来源主导不确定性估计。质量统计"
            "使用多个 CPU 进程并行编码；并行度仅用于缩短评测时间，不作为质量指标。"
        ),
        "",
        "## 3. 核心质量结果",
        "",
        "| 指标 | 32K 结果 |",
        "| --- | ---: |",
        f"| 文档数 | {summary['document_count']:,} |",
        f"| 字符数 | {summary['character_count']:,} |",
        f"| Token 数 | {summary['token_count']:,} |",
        f"| 字符/Token | {summary['characters_per_token']:.6f} |",
        f"| UTF-8 字节/Token | {summary['bytes_per_token']:.6f} |",
        f"| 文档级字符/Token P05 / P50 / P95 | {summary['p05_characters_per_token']:.3f} / {summary['p50_characters_per_token']:.3f} / {summary['p95_characters_per_token']:.3f} |",
        f"| 未知 Token | {summary['unknown_count']} |",
        f"| NFC 往返失败 | {summary['roundtrip_failures']} |",
        (
            f"| held-out 观测词表覆盖 | "
            f"{candidate['vocabulary_observation']['observed_token_ids']:,} / "
            f"{candidate['vocabulary_observation']['vocab_size']:,} "
            f"({candidate['vocabulary_observation']['heldout_vocabulary_coverage']:.2%}) |"
        ),
        "",
        "### 3.1 按语言",
        "",
        "| 语言 | 文档 | 字符 | Token | 字符/Token | 95% CI | 32K 相对 48K Token 增幅 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language, metrics in candidate["by_language"].items():
        increase = compare["by_language"][language]["candidate_token_increase_percent"]
        lines.append(
            f"| {language} | {metrics['document_count']:,} | "
            f"{metrics['character_count']:,} | {metrics['token_count']:,} | "
            f"{metrics['characters_per_token']:.4f} | "
            f"{metrics['characters_per_token_ci95_low']:.4f}–"
            f"{metrics['characters_per_token_ci95_high']:.4f} | {increase:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### 3.2 按内容类型",
            "",
            "| 内容类型 | 文档 | 字符/Token | 字节/Token | 未知 Token | 往返失败 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for content, metrics in candidate["by_content_type"].items():
        lines.append(
            f"| {content} | {metrics['document_count']:,} | "
            f"{metrics['characters_per_token']:.4f} | {metrics['bytes_per_token']:.4f} | "
            f"{metrics['unknown_count']} | {metrics['roundtrip_failures']} |"
        )
    lines.extend(
        [
            "",
            "### 3.3 按公开数据源",
            "",
            "| 来源 | 语言 | 内容 | 文档 | 字符/Token | 字节/Token |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    source_metadata = report["dataset"]["source_metadata"]
    for source, metrics in candidate["by_source"].items():
        metadata = source_metadata[source]
        lines.append(
            f"| {_source_label(source)} | {metadata['language']} | "
            f"{metadata['content_type']} | {metrics['document_count']:,} | "
            f"{metrics['characters_per_token']:.4f} | {metrics['bytes_per_token']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 鲁棒性探针",
            "",
            "| 探针 | 样本 | Token | 未知 Token | NFC 往返失败 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for suite, metrics in report["robustness"].items():
        lines.append(
            f"| {suite} | {metrics['sample_count']} | {metrics['token_count']} | "
            f"{metrics['unknown_count']} | {metrics['roundtrip_failures']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 结论与限制",
            "",
            "- 32K 正式 Tokenizer 在覆盖最终 100B 配方全部 20 个来源的合并评测集上实现零未知 Token、零 NFC 往返失败。",
            (
                f"- 相比 48K，32K 平均多使用 "
                f"{compare['overall']['candidate_token_increase_percent']:.2f}% Token；"
                "这是较小词表与模型侧效率之间的明确取舍。"
            ),
            "- 中文压缩率应优先看字符/Token，英文与代码可同时参考字符/Token和字节/Token。",
            "- 原 19 个 held-out 来源各 100 篇，DCLM 为 1,000 篇；该规模适合版本门禁和横向比较，但不能替代下游任务效果评测。",
            "",
            "## 6. 复现",
            "",
            "```bash",
            "source .venv/bin/activate",
            "python -m atomllm.tokenizer.technical_evaluation \\",
            "  --config configs/tokenizer/technical-evaluation-32k-v1.yaml \\",
            "  --overwrite",
            "```",
            "",
            "完整机器可读结果位于 "
            "`artifacts/tokenizer-technical-evaluations/atom-tokenizer-32k-v1/report.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(
    config_path: Path, project_root: Path, *, overwrite: bool
) -> dict[str, Any]:
    root = project_root.resolve()
    config_file = _resolve(root, config_path)
    config = load_config(config_file)
    candidate_dir = _resolve(root, config.candidate_tokenizer)
    comparison_dir = _resolve(root, config.comparison_tokenizer)
    snapshot_dir = _resolve(root, config.snapshot_dir)
    pretraining_plan_path = _resolve(root, config.pretraining_plan)
    output_dir = _resolve(root, config.output_dir)
    report_path = _resolve(root, config.report_path)
    if output_dir.exists() and not overwrite:
        raise TechnicalEvaluationError(
            f"output exists; pass --overwrite to replace it: {output_dir}"
        )
    documents, snapshot, heldout_path = _load_documents(snapshot_dir)
    candidate_tokenizer, candidate_manifest, candidate_manifest_path = (
        verify_tokenizer_directory(candidate_dir)
    )
    comparison_tokenizer, comparison_manifest, comparison_manifest_path = (
        verify_tokenizer_directory(comparison_dir)
    )
    _verify_lineage(candidate_manifest, snapshot)
    _verify_lineage(comparison_manifest, snapshot)
    try:
        pretraining_plan = yaml.safe_load(
            pretraining_plan_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise TechnicalEvaluationError("cannot read pretraining plan") from error
    if (
        not isinstance(pretraining_plan, dict)
        or pretraining_plan.get("total_target_tokens") != 100_000_000_000
    ):
        raise TechnicalEvaluationError("pretraining plan is not the 100B contract")
    heldout_source_count = len({document.source_id for document in documents})
    source_targets = pretraining_plan.get("source_target_tokens")
    if not isinstance(source_targets, dict):
        raise TechnicalEvaluationError("pretraining source targets are invalid")
    external_probe_metadata: dict[str, Any] = {}
    for index, probe in enumerate(config.external_probes):
        if probe.source_id not in source_targets:
            raise TechnicalEvaluationError(
                f"external probe is absent from pretraining plan: {probe.source_id}"
            )
        probe_documents, probe_metadata = _load_external_probe(
            root, probe, config.bootstrap_seed + index
        )
        documents.extend(probe_documents)
        external_probe_metadata[probe.source_id] = probe_metadata
    candidate_path = candidate_dir / "tokenizer.json"
    comparison_path = comparison_dir / "tokenizer.json"
    candidate_rows, candidate_frequencies = _evaluate_parallel(
        candidate_path, documents, config.quality_workers
    )
    comparison_rows, comparison_frequencies = _evaluate_parallel(
        comparison_path, documents, config.quality_workers
    )
    candidate_quality = _quality_report(
        candidate_rows,
        candidate_frequencies,
        candidate_manifest["vocab_size"],
        config.bootstrap_resamples,
        config.bootstrap_seed,
    )
    comparison_quality = _quality_report(
        comparison_rows,
        comparison_frequencies,
        comparison_manifest["vocab_size"],
        config.bootstrap_resamples,
        config.bootstrap_seed,
    )
    for quality in (candidate_quality, comparison_quality):
        if (
            quality["summary"]["unknown_count"]
            or quality["summary"]["roundtrip_failures"]
        ):
            raise TechnicalEvaluationError("tokenizer correctness gate failed")
    source_metadata: dict[str, dict[str, str]] = {}
    for document in documents:
        source_metadata.setdefault(
            document.source_id,
            {"language": document.language, "content_type": document.content_type},
        )
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "generated_at": generated_at,
        "config_sha256": _sha256(config_file),
        "candidate": {
            "artifact_id": candidate_manifest["artifact_id"],
            "manifest_sha256": _sha256(candidate_manifest_path),
            "tokenizer_sha256": _sha256(candidate_path),
            "tokenizer_bytes": candidate_path.stat().st_size,
            "vocab_size": candidate_manifest["vocab_size"],
            "algorithm": candidate_manifest["algorithm"]["model_type"],
        },
        "comparison_tokenizer": {
            "artifact_id": comparison_manifest["artifact_id"],
            "manifest_sha256": _sha256(comparison_manifest_path),
            "tokenizer_sha256": _sha256(comparison_path),
            "tokenizer_bytes": comparison_path.stat().st_size,
            "vocab_size": comparison_manifest["vocab_size"],
            "algorithm": comparison_manifest["algorithm"]["model_type"],
        },
        "dataset": {
            "pretraining_plan": config.pretraining_plan.as_posix(),
            "pretraining_plan_sha256": _sha256(pretraining_plan_path),
            "pretraining_target_tokens": pretraining_plan["total_target_tokens"],
            "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
            "heldout_sha256": _sha256(heldout_path),
            "heldout_training_disjoint_by_construction": True,
            "heldout_source_count": heldout_source_count,
            "external_probe_source_count": len(external_probe_metadata),
            "combined_source_count": len(source_metadata),
            "combined_document_count": len(documents),
            "documents_per_source": snapshot["heldout_documents_per_source"],
            "external_probe_metadata": external_probe_metadata,
            "source_metadata": dict(sorted(source_metadata.items())),
        },
        "methodology": {
            "quality_workers": config.quality_workers,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_seed": config.bootstrap_seed,
            "normalization": "NFC",
        },
        "quality": {
            "candidate": candidate_quality,
            "comparison": comparison_quality,
        },
        "comparison": _comparison(candidate_quality, comparison_quality),
        "robustness": _robustness(candidate_tokenizer, config.robustness_suites),
    }
    identity = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    report = {
        **payload,
        "evaluation_id": f"tokenizer-technical-eval-{digest[:12]}",
        "identity_sha256": digest,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        json_path = temporary / "report.json"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(config_file, temporary / "config.yaml")
        (temporary / "COMPLETED").write_text(
            f"{_sha256(json_path)}  report.json\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/technical-evaluation-32k-v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate(args.config, args.project_root, overwrite=args.overwrite)
    print(
        f"Tokenizer technical evaluation complete: {report['evaluation_id']}; "
        f"report={report['name']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
