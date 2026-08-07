#!/usr/bin/env python3
"""Migrate the pre-v1 local artifact layout to the versioned storage layout.

The migration is intentionally filesystem-local and idempotent.  It moves large
artifacts instead of copying them, rewrites metadata references, and updates the
WebDataset content digests after changing paths inside tar ``meta.json`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _same_file(left: Path, right: Path) -> bool:
    return (
        left.is_file()
        and right.is_file()
        and left.stat().st_size == right.stat().st_size
        and _sha256(left) == _sha256(right)
    )


def _move(src: Path, dst: Path) -> None:
    """Merge/move a path without clobbering a different existing artifact."""
    if src.is_symlink():
        # W&B's ``latest-run`` and debug links are disposable pointers.  The
        # actual run directories are moved separately; do not preserve links
        # whose relative targets would become invalid after relocation.
        src.unlink()
        return
    if not src.exists():
        return
    if not dst.exists() and not dst.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return
    if src.is_dir() and dst.is_dir():
        for child in list(src.iterdir()):
            _move(child, dst / child.name)
        src.rmdir()
        return
    if _same_file(src, dst):
        src.unlink()
        return
    # Shared W&B SDK directories contain process-global debug files.  Preserve
    # both versions under a deterministic legacy name when their contents differ.
    if "tracking" in dst.parts and dst.is_file() and src.is_file():
        renamed = dst.with_name(f"{dst.stem}-legacy-{_sha256(src)[:12]}{dst.suffix}")
        if not renamed.exists():
            shutil.move(str(src), str(renamed))
            return
        if _same_file(src, renamed):
            src.unlink()
            return
    raise RuntimeError(f"refusing to overwrite different artifact: {src} -> {dst}")


def _move_tree(src: Path, dst: Path) -> None:
    if src.exists() or src.is_symlink():
        _move(src, dst)


def _move_contents(src: Path, dst: Path) -> None:
    """Move children when the destination is intentionally inside ``src``."""
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in list(src.iterdir()):
        if child == dst:
            continue
        _move(child, dst / child.name)
    try:
        src.rmdir()
    except OSError:
        pass


def _remove_empty_tree(path: Path) -> None:
    if not path.is_dir():
        return
    for child in list(path.iterdir()):
        if child.is_dir() and not child.is_symlink():
            _remove_empty_tree(child)
    try:
        path.rmdir()
    except OSError:
        pass


def _replace_path(value: str, storage: Path, data: Path) -> str:
    """Translate both absolute legacy paths and dataset-relative references."""
    old = str(storage)
    new_data = str(data)
    # Older Hydra/W&B command strings occasionally rendered an empty path
    # component (``.../datasets//processed``).  Normalize that known local
    # prefix before applying the more specific dataset mappings below.
    normalized_value = value.replace(f"{old}/datasets//", f"{old}/datasets/")
    absolute = (
        (f"{old}/datasets/normalized/episodes/canary", f"{new_data}/canary/v1/normalized/episodes"),
        (f"{old}/datasets/normalized/screens/canary", f"{new_data}/canary/v1/normalized/screens"),
        (f"{old}/datasets/normalized/sources", f"{new_data}/registry/normalized/sources"),
        (f"{old}/datasets/normalized", f"{new_data}/registry/normalized"),
        (f"{old}/datasets/raw/license-records", f"{new_data}/registry/licenses"),
        (f"{old}/datasets/raw", f"{storage}/assets/sources"),
        (f"{old}/datasets/licenses", f"{new_data}/registry/licenses"),
        (f"{old}/datasets/text/canary-plans.json", f"{new_data}/canary/v1/text/plans.json"),
        (f"{old}/datasets/text", f"{new_data}/canary/v1/text"),
        (f"{old}/datasets/synthesized/canary", f"{new_data}/canary/v1/synthesized"),
        (
            f"{old}/datasets/synthesized/canary-utterances.jsonl",
            f"{new_data}/canary/v1/synthesized/utterances.jsonl",
        ),
        (f"{old}/datasets/synthesized", f"{new_data}/registry/synthesized"),
        (f"{old}/datasets/manifests/canary", f"{new_data}/canary/v1/manifests"),
        (f"{old}/datasets/manifests/pilot", f"{new_data}/pilot/v1/manifests"),
        (f"{old}/datasets/manifests", f"{new_data}/registry/manifests"),
        (f"{old}/datasets/staging/canary", f"{new_data}/canary/v1/shards/staging"),
        (f"{old}/datasets/staging/pilot", f"{new_data}/pilot/v1/shards/staging"),
        (f"{old}/datasets/processed/canary", f"{new_data}/canary/v1/shards/processed"),
        (f"{old}/datasets/processed/pilot", f"{new_data}/pilot/v1/shards/processed"),
        (f"{old}/datasets/processed", f"{new_data}/registry/processed"),
        (f"{old}/datasets/reports/canary", f"{new_data}/canary/v1/reports"),
        (f"{old}/datasets/reports", f"{new_data}/registry/reports"),
        (f"{old}/datasets/voices", f"{new_data}/registry/voices"),
        # W&B SDK metadata from older runs recorded the per-experiment SDK
        # directory.  It now lives in one shared tracking tree.
        (
            f"{old}/canary-run-300-continuous/runs/wandb",
            f"{storage}/tracking/wandb",
        ),
        (f"{old}/canary-run/runs/wandb", f"{storage}/tracking/wandb"),
        (
            f"{old}/direct-speech-overfit-run/runs/wandb",
            f"{storage}/tracking/wandb",
        ),
        (f"{old}/datasets", new_data),
        (f"{old}/models", f"{storage}/assets/models"),
        (f"{old}/vendor", f"{storage}/assets/vendor"),
        (f"{old}/canary-sources", f"{storage}/assets/sources/cache"),
        (f"{old}/canary-run-300-continuous", f"{storage}/experiments/canary/continuous-300"),
        (f"{old}/canary-run", f"{storage}/experiments/canary/default"),
        (
            f"{old}/direct-speech-overfit-run",
            f"{storage}/experiments/gates/direct-speech-overfit/default",
        ),
        (f"{old}/direct-speech-overfit", f"{new_data}/gates/direct-speech-overfit/v1"),
        (f"{old}/checkpoints", f"{storage}/checkpoints/smoke"),
        (f"{old}/logs", f"{storage}/runtime/logs"),
        (f"{old}/run", f"{storage}/runtime/sockets"),
        (f"{old}/runs/wandb", f"{storage}/tracking/wandb"),
    )
    for source, target in sorted(absolute, key=lambda item: len(item[0]), reverse=True):
        if normalized_value == source or normalized_value.startswith(source + "/"):
            suffix = normalized_value[len(source) :]
            # A few historical Hydra overrides contained ``datasets//...``.
            # Keep canonical absolute paths normalized without touching URL
            # schemes or arbitrary strings.
            if suffix.startswith("//"):
                suffix = suffix[1:]
            return target + suffix

    # Some SDK metadata stores the command-line override as a single string,
    # e.g. ``runtime.data_root=/old/root``.  Rewrite embedded path fragments too.
    embedded = normalized_value
    for source, target in sorted(absolute, key=lambda item: len(item[0]), reverse=True):
        embedded = embedded.replace(source, target)
    embedded = (
        embedded.replace(f"{storage}/pilot-data", f"{data}/")
        .replace(
            f"{storage}/canary-run-300-continuous", f"{storage}/experiments/canary/continuous-300"
        )
        .replace(f"{storage}/canary-run", f"{storage}/experiments/canary/default")
    )
    # The embedded replacement above can inherit a second slash from the
    # malformed legacy path.  Collapse it only immediately after known local
    # canonical roots.
    canonical_roots = {
        target.rstrip("/")
        for _, target in absolute
        if target.startswith("/")
    }
    for root in sorted(canonical_roots, key=len, reverse=True):
        embedded = embedded.replace(root + "//", root + "/")
    if embedded != value:
        return embedded

    relative = (
        ("normalized/episodes/canary", "canary/v1/normalized/episodes"),
        ("normalized/screens/canary", "canary/v1/normalized/screens"),
        ("normalized/sources", "registry/normalized/sources"),
        ("normalized/source-items", "registry/normalized/source-items"),
        ("synthesized/canary", "canary/v1/synthesized"),
        ("text/canary-plans.json", "canary/v1/text/plans.json"),
        ("manifests/canary", "canary/v1/manifests"),
        ("reports/canary", "canary/v1/reports"),
        ("voices/", "registry/voices/"),
        ("licenses/", "registry/licenses/"),
        ("raw/", f"{storage}/assets/sources/"),
        ("licenses/", f"{data}/registry/licenses/"),
    )
    for source, target in relative:
        if value == source or value.startswith(source + "/") or value.startswith(source):
            return target + value[len(source) :]
    return value


def _rewrite(value: Any, storage: Path, data: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite(item, storage, data) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, storage, data) for item in value]
    if isinstance(value, str):
        return _replace_path(value, storage, data)
    return value


def _rewrite_json(path: Path, storage: Path, data: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    rewritten = _rewrite(value, storage, data)
    if rewritten == value:
        return False
    temporary = path.with_name(f".{path.name}.migration-tmp")
    temporary.write_text(
        json.dumps(rewritten, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return True


def _rewrite_jsonl(path: Path, storage: Path, data: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    rewritten = [_rewrite(value, storage, data) for value in values]
    if rewritten == values:
        return False
    temporary = path.with_name(f".{path.name}.migration-tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n" for value in rewritten),
        encoding="utf-8",
    )
    temporary.replace(path)
    return True


def _sample_digest(members: dict[str, bytes], episode_id: str) -> str:
    digest = hashlib.sha256()
    for name in sorted(members):
        key = name[len(episode_id) + 1 :] if name.startswith(episode_id + ".") else name
        if key == "__key__":
            continue
        digest.update(key.encode())
        digest.update(members[name])
    return digest.hexdigest()


def _rewrite_shard(path: Path, storage: Path, data: Path) -> dict[str, str]:
    changed: dict[str, str] = {}
    temporary = path.with_name(f".{path.name}.migration-tmp")
    with tarfile.open(path, "r") as source, tarfile.open(temporary, "w") as target:
        for member in source:
            payload = source.extractfile(member).read() if member.isfile() else None
            if payload is not None and member.name.endswith(".meta.json"):
                try:
                    metadata = json.loads(payload.decode("utf-8"))
                    payload = (
                        json.dumps(_rewrite(metadata, storage, data), sort_keys=True) + "\n"
                    ).encode()
                except json.JSONDecodeError:
                    pass
            if payload is not None:
                member.size = len(payload)
                target.addfile(member, io.BytesIO(payload))
            else:
                target.addfile(member)
    temporary.replace(path)

    with tarfile.open(path, "r") as source:
        grouped: dict[str, dict[str, bytes]] = {}
        for member in source:
            if not member.isfile():
                continue
            episode_id = member.name.split(".", 1)[0]
            grouped.setdefault(episode_id, {})[member.name] = source.extractfile(member).read()
    for episode_id, members in grouped.items():
        changed[episode_id] = _sample_digest(members, episode_id)
    return changed


def _rewrite_shard_manifests(data: Path, storage: Path) -> int:
    updated = 0
    for path in data.rglob("*-manifest.jsonl"):
        if "shards" not in path.parts:
            continue
        digests: dict[str, str] = {}
        shard_dir = path.parent
        for shard in sorted(shard_dir.glob("*.tar")):
            digests.update(_rewrite_shard(shard, storage, data))
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            entry = _rewrite(entry, storage, data)
            if entry.get("episode_id") in digests:
                entry["content_sha256"] = digests[entry["episode_id"]]
            entries.append(entry)
        path.write_text(
            "".join(
                json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n" for entry in entries
            ),
            encoding="utf-8",
        )
        updated += 1
    return updated


def _move_legacy(storage: Path) -> None:
    data = storage / "datasets"
    assets = storage / "assets"
    registry = data / "registry"
    for src, dst in (
        (storage / "models", assets / "models"),
        (storage / "vendor", assets / "vendor"),
        (storage / "canary-sources", assets / "sources" / "cache"),
        (data / "raw" / "license-records", registry / "licenses"),
        (data / "raw" / "source-lock.json", registry / "source-lock.json"),
        (data / "raw", assets / "sources"),
        (data / "licenses", registry / "licenses"),
        (data / "normalized" / "sources", registry / "normalized" / "sources"),
        (
            data / "normalized" / "source-items.jsonl",
            registry / "normalized" / "source-items.jsonl",
        ),
        (
            data / "normalized" / "source-items.receipt.json",
            registry / "normalized" / "source-items.receipt.json",
        ),
        (
            data / "normalized" / "episodes" / "canary",
            data / "canary" / "v1" / "normalized" / "episodes",
        ),
        (
            data / "normalized" / "screens" / "canary",
            data / "canary" / "v1" / "normalized" / "screens",
        ),
        (data / "text" / "canary-plans.json", data / "canary" / "v1" / "text" / "plans.json"),
        (data / "synthesized" / "canary", data / "canary" / "v1" / "synthesized"),
        (
            data / "synthesized" / "canary-utterances.jsonl",
            data / "canary" / "v1" / "synthesized" / "utterances.jsonl",
        ),
        (data / "synthesized" / "cache", registry / "synthesis-cache"),
        (data / "manifests" / "canary", data / "canary" / "v1" / "manifests"),
        (data / "manifests" / "pilot", data / "pilot" / "v1" / "manifests"),
        (data / "staging" / "canary", data / "canary" / "v1" / "shards" / "staging"),
        (data / "staging" / "pilot", data / "pilot" / "v1" / "shards" / "staging"),
        (data / "processed" / "canary", data / "canary" / "v1" / "shards" / "processed"),
        (data / "processed" / "pilot", data / "pilot" / "v1" / "shards" / "processed"),
        (data / "reports" / "canary", data / "canary" / "v1" / "reports"),
        (data / "reports" / "canary-audit.json", data / "canary" / "v1" / "reports" / "audit.json"),
        (
            data / "reports" / "canary-codec-benchmark.json",
            data / "canary" / "v1" / "reports" / "codec-benchmark.json",
        ),
        (
            data / "reports" / "canary-encoded-report.json",
            data / "canary" / "v1" / "reports" / "encoded.json",
        ),
        (
            data / "reports" / "canary-manifest-report.json",
            data / "canary" / "v1" / "reports" / "manifest.json",
        ),
        (
            data / "reports" / "canary-mimi-decode.json",
            data / "canary" / "v1" / "reports" / "mimi-decode.json",
        ),
        (
            data / "reports" / "canary-readiness.json",
            data / "canary" / "v1" / "reports" / "readiness.json",
        ),
        (
            data / "reports" / "canary-synthesis-report.json",
            data / "canary" / "v1" / "reports" / "synthesis.json",
        ),
        (
            data / "reports" / "canary-text-report.json",
            data / "canary" / "v1" / "reports" / "text.json",
        ),
        (data / "reports" / "speech-smoke", registry / "reports" / "speech-smoke"),
        (data / "reports" / "fetch-report.json", registry / "reports" / "fetch-report.json"),
        (data / "reports" / "prepare-report.json", registry / "prepare.json"),
        (data / "voices", registry / "voices"),
        (
            storage / "canary-run-300-continuous",
            storage / "experiments" / "canary" / "continuous-300",
        ),
        (storage / "canary-run", storage / "experiments" / "canary" / "default"),
        (
            storage / "direct-speech-overfit-run",
            storage / "experiments" / "gates" / "direct-speech-overfit" / "default",
        ),
        (storage / "direct-speech-overfit", data / "gates" / "direct-speech-overfit" / "v1"),
        (storage / "logs", storage / "runtime" / "logs"),
        (storage / "run", storage / "runtime" / "sockets"),
        (storage / "runs" / "wandb", storage / "tracking" / "wandb"),
    ):
        _move_tree(src, dst)

    _move_contents(storage / "checkpoints", storage / "checkpoints" / "smoke")

    report_root = data / "canary" / "v1" / "reports"
    report_names = {
        "canary-audit.json": "audit.json",
        "canary-codec-benchmark.json": "codec-benchmark.json",
        "canary-encoded-report.json": "encoded.json",
        "canary-manifest-report.json": "manifest.json",
        "canary-mimi-decode.json": "mimi-decode.json",
        "canary-readiness.json": "readiness.json",
        "canary-synthesis-report.json": "synthesis.json",
        "canary-text-report.json": "text.json",
    }
    for old_name, new_name in report_names.items():
        old_report = report_root / old_name
        if old_report.is_file():
            _move_tree(old_report, report_root / new_name)

    # Remove only empty legacy containers; all non-empty history is retained in
    # the canonical tree or archive.
    for path in (
        data / "raw",
        data / "licenses",
        data / "normalized",
        data / "manifests",
        data / "processed",
        data / "reports",
        data / "staging",
        data / "synthesized",
        data / "text",
        storage / "runs",
    ):
        _remove_empty_tree(path)

    # Older runs kept SDK files under each experiment.  Consolidate those files
    # into the shared tracking volume while retaining evaluation/checkpoint data.
    for path in sorted((storage / "experiments").glob("**/runs/wandb")):
        _move_tree(path, storage / "tracking" / "wandb")

    archive = storage / "archive"
    for path in sorted(storage.glob("direct-speech-overfit-control*")):
        _move_tree(path, archive / path.name)

    gate_root = data / "gates" / "direct-speech-overfit" / "v1"
    for name in ("staging", "processed"):
        source = gate_root / name
        if source.is_dir():
            destination = gate_root / "shards" / name
            _move_tree(source, destination)
    for name in ("staging", "processed"):
        destination = gate_root / "shards" / name
        flat_train = destination / "train-000000.tar"
        flat_manifest = destination / "train-manifest.jsonl"
        if flat_train.exists() or flat_manifest.exists():
            _move_tree(flat_train, destination / "train" / flat_train.name)
            _move_tree(flat_manifest, destination / "train" / flat_manifest.name)


def _update_derived_hashes(storage: Path) -> None:
    data = storage / "datasets"
    registry = data / "registry"
    voice_registry = registry / "voices" / "registry.json"
    if voice_registry.is_file():
        value = json.loads(voice_registry.read_text(encoding="utf-8"))
        value["registry_sha256"] = _stable_hash(
            {key: item for key, item in value.items() if key != "registry_sha256"}
        )
        voice_registry.write_text(
            json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8"
        )
    for dataset in ("canary", "pilot"):
        root = data / dataset / "v1"
        manifest = root / "manifests" / "episodes.jsonl"
        if not manifest.is_file():
            continue
        manifest_hash = _sha256(manifest)
        for name in ("manifest.json", "audit.json", "mimi-decode.json"):
            report = root / "reports" / name
            if report.is_file():
                value = json.loads(report.read_text(encoding="utf-8"))
                if "manifest_sha256" in value:
                    value["manifest_sha256"] = manifest_hash
                    report.write_text(
                        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
        hash_report = root / "reports" / "manifest-hash.json"
        if hash_report.is_file():
            hash_report.write_text(
                json.dumps({"sha256": manifest_hash}, indent=2) + "\n", encoding="utf-8"
            )
        synthesis = root / "reports" / "synthesis.json"
        utterances = root / "synthesized" / "utterances.jsonl"
        if synthesis.is_file() and utterances.is_file():
            value = json.loads(synthesis.read_text(encoding="utf-8"))
            value["manifest_sha256"] = _sha256(utterances)
            synthesis.write_text(
                json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def _rewrite_all_metadata(storage: Path) -> tuple[int, int]:
    data = storage / "datasets"
    changed_json = changed_jsonl = 0
    # Binary model/vendor artifacts are never parsed; JSON metadata under the
    # shared tracking directory is safe to rewrite and keeps historical run
    # pointers consistent with the canonical layout.
    skip = {storage / "assets"}
    for path in storage.rglob("*"):
        if not path.is_file() or any(parent in skip for parent in path.parents):
            continue
        if path.suffix == ".json":
            changed_json += int(_rewrite_json(path, storage, data))
        elif path.suffix == ".jsonl":
            changed_jsonl += int(_rewrite_jsonl(path, storage, data))
    return changed_json, changed_jsonl


def migrate(storage: Path) -> dict[str, Any]:
    storage = storage.expanduser().resolve()
    _move_legacy(storage)
    changed_json, changed_jsonl = _rewrite_all_metadata(storage)
    shard_manifests = _rewrite_shard_manifests(storage / "datasets", storage)
    _update_derived_hashes(storage)
    return {
        "storage_root": str(storage),
        "changed_json": changed_json,
        "changed_jsonl": changed_jsonl,
        "rewritten_shard_manifests": shard_manifests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default="~/latentloop-data")
    args = parser.parse_args()
    print(json.dumps(migrate(Path(args.storage_root)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
