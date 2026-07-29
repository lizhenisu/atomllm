"""Strict schemas for source registration and canonical data records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml


SCHEMA_VERSION = 1
PRIVACY_ACTION = "warn"
VALID_CONTENT_TYPES = frozenset(
    {"general", "code", "math", "science", "encyclopedia", "conversation"}
)
VALID_PROVIDERS = frozenset({"http", "huggingface", "local", "synthetic"})

_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_DOCUMENT_ID_PATTERN = re.compile(r"^doc-[0-9a-f]{64}$")
_LANGUAGE_PATTERN = re.compile(r"^(?:code|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*|und)$")
_WARNING_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FLOATING_REVISIONS = frozenset({"latest", "main", "master", "head"})


class DataSchemaError(ValueError):
    """Raised when source metadata or a canonical document violates the contract."""


def _require_exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise DataSchemaError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise DataSchemaError(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataSchemaError(f"{field_name} must be a non-empty string")
    return value


def _require_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DataSchemaError(f"{field_name} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise DataSchemaError(f"{field_name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise DataSchemaError(f"{field_name} must not contain duplicates")
    return tuple(value)


def _validate_schema_version(value: Any, context: str) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise DataSchemaError(
            f"{context} schema_version must be {SCHEMA_VERSION}, got {value!r}"
        )


def _validate_json_value(value: Any, path: str = "metadata") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataSchemaError(f"{path} must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DataSchemaError(f"{path} object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise DataSchemaError(f"{path} contains a non-JSON value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Acquisition:
    provider: str
    location: str
    revision: str
    expected_sha256: str | None

    @classmethod
    def from_mapping(cls, data: Any) -> Acquisition:
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise DataSchemaError("acquisition must be a mapping with string keys")
        _require_exact_keys(
            data,
            {"provider", "location", "revision", "expected_sha256"},
            "acquisition",
        )

        provider = _require_non_empty_string(data["provider"], "acquisition.provider")
        if provider not in VALID_PROVIDERS:
            choices = ", ".join(sorted(VALID_PROVIDERS))
            raise DataSchemaError(f"acquisition.provider must be one of: {choices}")

        location = _require_non_empty_string(data["location"], "acquisition.location")
        revision = _require_non_empty_string(data["revision"], "acquisition.revision")
        if revision.lower() in _FLOATING_REVISIONS:
            raise DataSchemaError("acquisition.revision must be immutable")

        expected_sha256 = data["expected_sha256"]
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise DataSchemaError(
                "acquisition.expected_sha256 must be null or 64 lowercase hex digits"
            )

        if provider == "local":
            local_path = Path(location)
            if local_path.is_absolute() or ".." in local_path.parts:
                raise DataSchemaError(
                    "local acquisition.location must be a safe relative path"
                )
        elif provider in {"http", "huggingface"}:
            parsed = urlsplit(location)
            if parsed.username is not None or parsed.password is not None:
                raise DataSchemaError(
                    "acquisition.location must not contain credentials"
                )
        elif provider == "synthetic" and not location.startswith("synthetic://"):
            raise DataSchemaError(
                "synthetic acquisition.location must use synthetic://"
            )

        return cls(
            provider=provider,
            location=location,
            revision=revision,
            expected_sha256=expected_sha256,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "location": self.location,
            "revision": self.revision,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True, slots=True)
class DataSource:
    source_id: str
    name: str
    version: str
    license: str
    homepage: str
    languages: tuple[str, ...]
    content_types: tuple[str, ...]
    data_format: str
    enabled: bool
    acquisition: Acquisition

    @classmethod
    def from_mapping(cls, data: Any) -> DataSource:
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise DataSchemaError("source must be a mapping with string keys")
        _require_exact_keys(
            data,
            {
                "source_id",
                "name",
                "version",
                "license",
                "homepage",
                "languages",
                "content_types",
                "data_format",
                "enabled",
                "acquisition",
            },
            "source",
        )

        source_id = _require_non_empty_string(data["source_id"], "source_id")
        if _SOURCE_ID_PATTERN.fullmatch(source_id) is None:
            raise DataSchemaError(
                "source_id must contain lowercase letters, digits, and hyphens"
            )

        name = _require_non_empty_string(data["name"], "name")
        version = _require_non_empty_string(data["version"], "version")
        if version.lower() in _FLOATING_REVISIONS:
            raise DataSchemaError("version must be immutable")
        license_name = _require_non_empty_string(data["license"], "license")
        homepage = _require_non_empty_string(data["homepage"], "homepage")
        parsed_homepage = urlsplit(homepage)
        if parsed_homepage.username is not None or parsed_homepage.password is not None:
            raise DataSchemaError("homepage must not contain credentials")
        languages = _require_string_list(data["languages"], "languages")
        if any(_LANGUAGE_PATTERN.fullmatch(item) is None for item in languages):
            raise DataSchemaError("languages contains an invalid language tag")

        content_types = _require_string_list(data["content_types"], "content_types")
        invalid_types = sorted(set(content_types) - VALID_CONTENT_TYPES)
        if invalid_types:
            raise DataSchemaError(
                f"content_types contains unsupported values: {', '.join(invalid_types)}"
            )

        data_format = _require_non_empty_string(data["data_format"], "data_format")
        enabled = data["enabled"]
        if type(enabled) is not bool:
            raise DataSchemaError("enabled must be a boolean")

        return cls(
            source_id=source_id,
            name=name,
            version=version,
            license=license_name,
            homepage=homepage,
            languages=languages,
            content_types=content_types,
            data_format=data_format,
            enabled=enabled,
            acquisition=Acquisition.from_mapping(data["acquisition"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "homepage": self.homepage,
            "languages": list(self.languages),
            "content_types": list(self.content_types),
            "data_format": self.data_format,
            "enabled": self.enabled,
            "acquisition": self.acquisition.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    schema_version: int
    privacy_action: str
    sources: tuple[DataSource, ...]

    @classmethod
    def from_mapping(cls, data: Any) -> SourceRegistry:
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise DataSchemaError("source registry must be a mapping with string keys")
        _require_exact_keys(
            data,
            {"schema_version", "privacy_action", "sources"},
            "source registry",
        )
        _validate_schema_version(data["schema_version"], "source registry")

        privacy_action = data["privacy_action"]
        if privacy_action != PRIVACY_ACTION:
            raise DataSchemaError("privacy_action must be 'warn'")

        raw_sources = data["sources"]
        if not isinstance(raw_sources, list):
            raise DataSchemaError("sources must be a list")
        sources = tuple(DataSource.from_mapping(item) for item in raw_sources)
        source_ids = [source.source_id for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise DataSchemaError("source_id values must be unique")

        return cls(
            schema_version=SCHEMA_VERSION,
            privacy_action=PRIVACY_ACTION,
            sources=sources,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "privacy_action": self.privacy_action,
            "sources": [source.to_mapping() for source in self.sources],
        }


def load_source_registry(path: str | Path) -> SourceRegistry:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"source registry not found: {registry_path}")
    try:
        raw_data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise DataSchemaError(f"invalid source registry YAML: {error}") from error
    return SourceRegistry.from_mapping(raw_data)


def make_document_id(source_id: str, source_record_id: str) -> str:
    validated_source_id = _require_non_empty_string(source_id, "source_id")
    if _SOURCE_ID_PATTERN.fullmatch(validated_source_id) is None:
        raise DataSchemaError(
            "source_id must contain lowercase letters, digits, and hyphens"
        )
    validated_record_id = _require_non_empty_string(
        source_record_id, "source_record_id"
    )
    identity = f"{validated_source_id}\0{validated_record_id}".encode()
    return f"doc-{hashlib.sha256(identity).hexdigest()}"


def _validate_warning_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DataSchemaError(f"{field_name} must be a list")
    if not all(
        isinstance(item, str) and _WARNING_PATTERN.fullmatch(item) is not None
        for item in value
    ):
        raise DataSchemaError(
            f"{field_name} must contain lowercase warning identifiers"
        )
    if len(value) != len(set(value)):
        raise DataSchemaError(f"{field_name} must not contain duplicates")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    schema_version: int
    document_id: str
    source_id: str
    source_record_id: str
    text: str
    language: str
    content_type: str
    privacy_warnings: tuple[str, ...]
    quality_warnings: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: Any) -> CanonicalDocument:
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise DataSchemaError("document must be a mapping with string keys")
        _require_exact_keys(
            data,
            {
                "schema_version",
                "document_id",
                "source_id",
                "source_record_id",
                "text",
                "language",
                "content_type",
                "privacy_warnings",
                "quality_warnings",
                "metadata",
            },
            "document",
        )
        _validate_schema_version(data["schema_version"], "document")

        source_id = _require_non_empty_string(data["source_id"], "source_id")
        source_record_id = _require_non_empty_string(
            data["source_record_id"], "source_record_id"
        )
        document_id = _require_non_empty_string(data["document_id"], "document_id")
        if _DOCUMENT_ID_PATTERN.fullmatch(document_id) is None:
            raise DataSchemaError("document_id must use the doc-<sha256> format")
        expected_id = make_document_id(source_id, source_record_id)
        if document_id != expected_id:
            raise DataSchemaError("document_id does not match source identity")

        text = _require_non_empty_string(data["text"], "text")
        language = _require_non_empty_string(data["language"], "language")
        if _LANGUAGE_PATTERN.fullmatch(language) is None:
            raise DataSchemaError("language must be a valid language tag or 'und'")

        content_type = _require_non_empty_string(data["content_type"], "content_type")
        if content_type not in VALID_CONTENT_TYPES:
            choices = ", ".join(sorted(VALID_CONTENT_TYPES))
            raise DataSchemaError(f"content_type must be one of: {choices}")

        privacy_warnings = _validate_warning_list(
            data["privacy_warnings"], "privacy_warnings"
        )
        quality_warnings = _validate_warning_list(
            data["quality_warnings"], "quality_warnings"
        )

        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            raise DataSchemaError("metadata must be a JSON object")
        _validate_json_value(metadata)

        return cls(
            schema_version=SCHEMA_VERSION,
            document_id=document_id,
            source_id=source_id,
            source_record_id=source_record_id,
            text=text,
            language=language,
            content_type=content_type,
            privacy_warnings=privacy_warnings,
            quality_warnings=quality_warnings,
            metadata=dict(metadata),
        )

    @classmethod
    def from_json_line(cls, line: str) -> CanonicalDocument:
        try:
            raw_data = json.loads(line)
        except json.JSONDecodeError as error:
            raise DataSchemaError(f"invalid document JSON: {error.msg}") from error
        return cls.from_mapping(raw_data)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_record_id": self.source_record_id,
            "text": self.text,
            "language": self.language,
            "content_type": self.content_type,
            "privacy_warnings": list(self.privacy_warnings),
            "quality_warnings": list(self.quality_warnings),
            "metadata": dict(self.metadata),
        }

    def to_json_line(self) -> str:
        return json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
