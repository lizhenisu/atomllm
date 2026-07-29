"""Safely migrate committed public-token groups to an append-only data plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from atomllm.data.public_pretraining_plan import load_plan
from atomllm.data.public_tokenizer_corpus import load_config as load_source_registry
from atomllm.training.public_token_shards import (
    _canonical_json,
    _source_priority,
)


class PublicPlanExtensionError(RuntimeError):
    """Raised when an attempted plan migration is not provably append-only."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sources(plan: Any, root: Path) -> tuple[Any, ...]:
    registry = load_source_registry(root / plan.source_registry)
    return (*registry.sources, *plan.supplemental_sources)


def _group_order(sources: tuple[Any, ...], group: str) -> list[str]:
    return [
        source.source_id
        for source in sorted(
            (source for source in sources if source.language == group),
            key=_source_priority,
        )
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicPlanExtensionError(message)


def _load_completed_group(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    completed_path = directory / "COMPLETED"
    _require(
        manifest_path.is_file() and completed_path.is_file(),
        f"completed group is missing: {directory.name}",
    )
    _require(
        completed_path.read_text(encoding="utf-8")
        == f"{_sha256(manifest_path)}  manifest.json\n",
        f"completed marker is invalid: {directory.name}",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    _require(
        isinstance(shards, list) and shards,
        f"group has no committed shards: {directory.name}",
    )
    for shard in shards:
        for field in ("token_file", "index_file", "digest_file"):
            metadata = shard.get(field)
            _require(
                isinstance(metadata, dict)
                and isinstance(metadata.get("name"), str)
                and (directory / metadata["name"]).is_file()
                and (directory / metadata["name"]).stat().st_size
                == metadata.get("size_bytes"),
                f"committed shard file is missing or truncated: "
                f"{directory.name}/{field}",
            )
    return manifest


def migrate_plan_extension(
    *,
    old_plan_path: Path,
    new_plan_path: Path,
    group_root: Path,
    project_root: Path = Path("."),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate one exhausted group and unchanged completed groups without rewriting shards."""
    root = project_root.resolve()
    old_path = (root / old_plan_path).resolve()
    new_path = (root / new_plan_path).resolve()
    groups = (root / group_root).resolve()
    _require(
        all(path.is_relative_to(root) for path in (old_path, new_path, groups)),
        "migration paths must remain inside project root",
    )
    old_sha = _sha256(old_path)
    new_sha = _sha256(new_path)
    old_plan = load_plan(old_path, project_root=root)
    new_plan = load_plan(new_path, project_root=root)
    _require(
        new_plan.parent_plan_sha256 == old_sha,
        "new plan does not declare the current plan as its parent",
    )
    for field in (
        "total_target_tokens",
        "language_target_tokens",
        "training_split",
        "validation_status",
        "sequence_length",
        "shard_token_capacity",
        "token_dtype",
        "document_boundary_token",
    ):
        _require(
            getattr(old_plan, field) == getattr(new_plan, field),
            f"plan extension changes immutable field: {field}",
        )

    old_sources = _sources(old_plan, root)
    new_sources = _sources(new_plan, root)
    old_by_id = {source.source_id: source for source in old_sources}
    new_by_id = {source.source_id: source for source in new_sources}
    added_ids = sorted(set(new_by_id) - set(old_by_id))
    _require(len(added_ids) == 1, "plan extension must append exactly one source")
    added_id = added_ids[0]
    _require(
        new_by_id[added_id].language == "en",
        "supplemental source must fill the English group",
    )
    _require(
        not (set(old_by_id) - set(new_by_id)),
        "plan extension removes an existing source",
    )
    changed_ids = sorted(
        source_id
        for source_id in old_by_id
        if old_plan.source_target_tokens[source_id]
        != new_plan.source_target_tokens[source_id]
    )
    _require(
        len(changed_ids) == 1,
        "plan extension must revise exactly one exhausted source target",
    )
    exhausted_id = changed_ids[0]
    reduction = (
        old_plan.source_target_tokens[exhausted_id]
        - new_plan.source_target_tokens[exhausted_id]
    )
    _require(reduction > 0, "existing source target must only be reduced")
    _require(
        reduction == new_plan.source_target_tokens[added_id],
        "supplemental target must exactly replace the exhausted shortfall",
    )
    for source_id in set(old_by_id) - {exhausted_id}:
        _require(
            old_plan.source_target_tokens[source_id]
            == new_plan.source_target_tokens[source_id],
            f"unrelated source target changed: {source_id}",
        )

    migration_dir = (
        groups / "_plan-migrations" / f"{old_sha[:12]}-to-{new_sha[:12]}"
    )
    receipt_path = migration_dir / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _require(
            receipt.get("old_plan_sha256") == old_sha
            and receipt.get("new_plan_sha256") == new_sha,
            "existing migration receipt does not match requested plans",
        )
        return receipt
    _require(not migration_dir.exists(), "incomplete migration directory already exists")

    en_state_path = groups / "en" / "state.json"
    _require(en_state_path.is_file(), "English resume state is missing")
    state = json.loads(en_state_path.read_text(encoding="utf-8"))
    _require(
        state.get("identity", {}).get("plan_sha256") == old_sha,
        "English state is not bound to the parent plan",
    )
    old_en_order = _group_order(old_sources, "en")
    new_en_order = _group_order(new_sources, "en")
    _require(
        new_en_order == [*old_en_order, added_id],
        "supplemental source is not append-only in English source order",
    )
    _require(state.get("source_order") == old_en_order, "English source order drifted")
    _require(
        state.get("source_index") == len(old_en_order),
        "English state is not positioned after all parent sources",
    )
    _require(
        state.get("source_exhausted", {}).get(exhausted_id) is True,
        "revised source is not recorded as exhausted",
    )
    actual_tokens = state.get("source_content_tokens", {}).get(exhausted_id)
    _require(
        actual_tokens == new_plan.source_target_tokens[exhausted_id],
        "new exhausted-source target does not equal committed content",
    )
    _require(
        state.get("carried_shortfall_tokens")
        == new_plan.source_target_tokens[added_id],
        "committed shortfall does not equal supplemental target",
    )

    completed: dict[str, dict[str, Any]] = {}
    for group in ("code", "zh-Hans"):
        manifest = _load_completed_group(groups / group)
        _require(
            manifest.get("identity", {}).get("plan_sha256") == old_sha,
            f"{group} manifest is not bound to the parent plan",
        )
        targets = manifest.get("source_target_tokens")
        _require(isinstance(targets, dict), f"{group} source targets are missing")
        _require(
            all(
                new_plan.source_target_tokens.get(source_id) == target
                for source_id, target in targets.items()
            ),
            f"{group} source targets changed in extension",
        )
        completed[group] = manifest

    validation = {
        "schema_version": 1,
        "migration": "append-one-public-source-v1",
        "dry_run": dry_run,
        "old_plan_sha256": old_sha,
        "new_plan_sha256": new_sha,
        "exhausted_source_id": exhausted_id,
        "exhausted_source_content_tokens": actual_tokens,
        "supplemental_source_id": added_id,
        "supplemental_target_tokens": new_plan.source_target_tokens[added_id],
        "preserved_english_shards": len(state["shards"]),
    }
    if dry_run:
        return validation

    migration_dir.mkdir(parents=True)
    shutil.copy2(old_path, migration_dir / "parent-plan.yaml")
    shutil.copy2(new_path, migration_dir / "extended-plan.yaml")
    shutil.copy2(en_state_path, migration_dir / "en-state.before.json")
    for group in completed:
        shutil.copy2(
            groups / group / "manifest.json",
            migration_dir / f"{group}-manifest.before.json",
        )
        shutil.copy2(
            groups / group / "COMPLETED",
            migration_dir / f"{group}-COMPLETED.before",
        )

    try:
        state["identity"]["plan_sha256"] = new_sha
        state["source_order"] = new_en_order
        state["source_records_seen"][added_id] = 0
        state["source_content_tokens"][added_id] = 0
        state["source_documents"][added_id] = 0
        state["source_effective_target_tokens"][exhausted_id] = (
            new_plan.source_target_tokens[exhausted_id]
        )
        # The extended plan adopts the committed amount as this source's new
        # target. It is therefore complete under the new plan; retaining the
        # parent plan's exhausted flag would make the completed group invalid.
        state["source_exhausted"][exhausted_id] = False
        state["carried_shortfall_tokens"] = 0
        _write_json(en_state_path, state)

        updated_manifest_hashes = {}
        for group, manifest in completed.items():
            manifest["identity"]["plan_sha256"] = new_sha
            manifest["dataset_id"] = (
                f"public-token-group-{group}-"
                f"{hashlib.sha256(_canonical_json(manifest['identity']).encode()).hexdigest()[:12]}"
            )
            manifest_path = groups / group / "manifest.json"
            _write_json(manifest_path, manifest)
            manifest_sha = _sha256(manifest_path)
            completed_path = groups / group / "COMPLETED"
            completed_path.write_text(
                f"{manifest_sha}  manifest.json\n", encoding="utf-8"
            )
            _require(
                _load_completed_group(groups / group).get("identity")
                == manifest["identity"],
                f"{group} identity update did not persist",
            )
            updated_manifest_hashes[group] = manifest_sha

        migrated_state = json.loads(en_state_path.read_text(encoding="utf-8"))
        _require(
            migrated_state["identity"]["plan_sha256"] == new_sha,
            "English state migration did not persist",
        )
        receipt = {
            **validation,
            "dry_run": False,
            "updated_completed_manifest_sha256": updated_manifest_hashes,
            "en_state_sha256": _sha256(en_state_path),
        }
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException:
        shutil.copy2(migration_dir / "en-state.before.json", en_state_path)
        for group in completed:
            shutil.copy2(
                migration_dir / f"{group}-manifest.before.json",
                groups / group / "manifest.json",
            )
            shutil.copy2(
                migration_dir / f"{group}-COMPLETED.before",
                groups / group / "COMPLETED",
            )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--new-plan", type=Path, required=True)
    parser.add_argument("--group-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    receipt = migrate_plan_extension(
        old_plan_path=args.old_plan,
        new_plan_path=args.new_plan,
        group_root=args.group_root,
        project_root=args.project_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
