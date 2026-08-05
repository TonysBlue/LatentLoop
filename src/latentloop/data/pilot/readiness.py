from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from latentloop.config import ProjectConfig
from latentloop.data.pilot.common import SPLITS, read_json, read_jsonl, write_json


def _required(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_file():
        missing.append(f"{label}: {path}")


def check_e2_readiness(
    root: str | Path,
    *,
    config: ProjectConfig,
    dataset: str = "pilot",
    require_checkpoint: str | Path | None = None,
    require_encoded: bool = True,
) -> dict[str, Any]:
    """Fail-closed machine check immediately before launching E2 training."""
    root = Path(root).expanduser().resolve()
    missing: list[str] = []
    invalid: list[str] = []
    audit_path = root / "reports" / f"{dataset}-audit.json"
    _required(audit_path, "audit report", missing)
    if audit_path.is_file() and not read_json(audit_path).get("passed"):
        invalid.append(f"audit report is not passed: {audit_path}")
    manifest_paths: dict[str, Path] = {}
    shard_paths: dict[str, list[Path]] = {}
    for split in SPLITS:
        manifest = root / "manifests" / dataset / f"{split}.jsonl"
        manifest_paths[split] = manifest
        _required(manifest, f"{split} manifest", missing)
        processed = root / "processed" / dataset / split
        shards = sorted(processed.glob(f"{split}-*.tar"))
        shard_paths[split] = shards
        if require_encoded:
            if not shards:
                missing.append(f"{split} processed shards: {processed / (split + '-*.tar')}")
            _required(
                processed / f"{split}-manifest.jsonl",
                f"{split} processed manifest",
                missing,
            )
    if require_encoded:
        for split in SPLITS:
            if not manifest_paths[split].is_file():
                continue
            source_entries = read_jsonl(manifest_paths[split])
            processed_manifest = root / "processed" / dataset / split / f"{split}-manifest.jsonl"
            if processed_manifest.is_file():
                encoded_entries = read_jsonl(processed_manifest)
                if {entry["episode_id"] for entry in source_entries} != {
                    entry["episode_id"] for entry in encoded_entries
                }:
                    invalid.append(f"{split} processed manifest does not match source manifest")
                if any(not entry.get("speech_codes_encoded") for entry in encoded_entries):
                    invalid.append(f"{split} processed manifest contains unencoded speech")
    mimi_reports = {}
    for split in (dataset,):
        report = root / "reports" / f"{split}-mimi-decode.json"
        _required(report, f"{split} Mimi decode report", missing)
        if report.is_file():
            mimi_reports[split] = read_json(report)
    if require_checkpoint:
        checkpoint = Path(require_checkpoint).expanduser()
        _required(checkpoint, "initial checkpoint", missing)
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if payload.get("format_version") not in {2, 3}:
                    invalid.append("initial checkpoint must use format version 2 or 3")
                metadata = payload.get("metadata", {})
                if metadata.get("codec_id") != config.data.codec_id:
                    invalid.append("initial checkpoint codec_id differs from E2 config")
                if metadata.get("codec_revision") != config.data.codec_revision:
                    invalid.append("initial checkpoint codec_revision differs from E2 config")
                if metadata.get("codec_weight_hash") != config.data.codec_weight_hash:
                    invalid.append("initial checkpoint codec weight hash differs from E2 config")
                state = payload.get("model", {})
                if not isinstance(state, dict) or not state:
                    invalid.append("initial checkpoint has no model state")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                invalid.append(f"initial checkpoint cannot be inspected: {error}")
    disk = shutil.disk_usage(root)
    result = {
        "dataset": dataset,
        "root": str(root),
        "passed": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "manifests": {split: str(path) for split, path in manifest_paths.items()},
        "shards": {split: [str(path) for path in paths] for split, paths in shard_paths.items()},
        "mimi_reports": mimi_reports,
        "free_disk_bytes": disk.free,
    }
    write_json(root / "reports" / f"{dataset}-readiness.json", result)
    if not result["passed"]:
        problems = "; ".join(missing + invalid)
        raise ValueError(f"E2 training readiness failed: {problems}")
    return result
