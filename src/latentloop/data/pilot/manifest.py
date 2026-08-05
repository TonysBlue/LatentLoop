from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from latentloop.data.pilot.audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    align_up,
    read_mono,
    write_flac,
)
from latentloop.data.pilot.common import (
    CATEGORIES,
    LANGUAGES,
    SPLITS,
    ensure_tree,
    read_json,
    read_jsonl,
    relative_to_root,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from latentloop.data.pilot.spec import dataset_spec


def _run_inventory_adapter(command: str, root: Path) -> None:
    output = root / "normalized" / "source-items.jsonl"
    receipt_path = root / "normalized" / "source-items.receipt.json"
    fetch_report = root / "reports" / "fetch-report.json"
    recipe = {
        "operation": "index-and-normalize",
        "fetch_report_sha256": sha256_file(fetch_report),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "format": "FLAC/PCM16",
        "adapter": command,
    }
    recipe_hash = stable_hash(recipe)
    if output.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("recipe_sha256") == recipe_hash and receipt.get(
            "inventory_sha256"
        ) == sha256_file(output):
            return
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(
            {
                **recipe,
                "raw_root": str(root / "raw"),
                "licenses_root": str(root / "licenses"),
                "output": str(output),
            },
            request_file,
        )
        request_file.flush()
        completed = subprocess.run(
            [*shlex.split(command), "--request", request_file.name, "--output", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"normalization adapter failed: {(completed.stderr or completed.stdout)[-1000:]}"
        )
    if not output.is_file():
        raise RuntimeError("normalization adapter did not create source-items.jsonl")
    write_json(
        receipt_path,
        {"recipe_sha256": recipe_hash, "inventory_sha256": sha256_file(output)},
    )


def _load_assets(
    root: Path, dataset: str
) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]
]:
    source_path = root / "normalized" / "source-items.jsonl"
    plans_path = root / "text" / f"{dataset}-plans.json"
    synthesis_path = root / "synthesized" / f"{dataset}-utterances.jsonl"
    voices_path = root / "voices" / "registry.json"
    for path in (source_path, plans_path, synthesis_path, voices_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Pilot asset is absent: {path}")
    source_items = read_jsonl(source_path)
    for item in source_items:
        audio = Path(item["audio"])
        if not audio.is_absolute():
            audio = root / audio
        if not audio.is_file() or sha256_file(audio) != item.get("audio_sha256"):
            raise ValueError(f"source inventory audio hash mismatch: {item.get('source_item_id')}")
    receipt_rows = read_jsonl(synthesis_path)
    for item in receipt_rows:
        audio = Path(item["audio"])
        if not audio.is_absolute():
            audio = root / audio
        if not audio.is_file() or sha256_file(audio) != item.get("audio_sha256"):
            raise ValueError(
                "synthesis receipt audio hash mismatch: "
                f"{item.get('plan_id') or item.get('source_item_id')}/{item.get('turn_id')}"
            )
    receipts = {
        (str(item.get("plan_id") or item.get("source_item_id")), str(item["turn_id"])): item
        for item in receipt_rows
    }
    if len(receipts) != len(receipt_rows):
        raise ValueError("synthesis receipt keys are duplicated")
    registry = read_json(voices_path)
    registry_hash = registry.get("registry_sha256")
    expected_registry = stable_hash(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    if registry_hash != expected_registry:
        raise ValueError("voice registry hash is stale")
    return source_items, read_json(plans_path), receipts, registry


def _excluded(root: Path, dataset: str, fixture: bool) -> tuple[set[str], set[str]]:
    if dataset == "canary":
        return set(), set()
    path = root / "manifests" / "canary" / "episodes.jsonl"
    if not path.is_file():
        raise ValueError("build and audit Canary before constructing the Pilot dataset")
    audit_path = root / "reports" / "canary-audit.json"
    if not audit_path.is_file():
        raise ValueError("audit Canary before constructing the Pilot dataset")
    audit = read_json(audit_path)
    if not audit.get("passed") or bool(audit.get("fixture")) != fixture:
        raise ValueError("Canary audit mode does not match the requested Pilot build")
    if audit.get("manifest_sha256") != sha256_file(path):
        raise ValueError("Canary manifest changed after its audit")
    plan_ids: set[str] = set()
    source_ids: set[str] = set()
    for record in read_jsonl(path):
        if record.get("plan_id"):
            plan_ids.add(str(record["plan_id"]))
        if record.get("category") in {"public_speech", "adjacent_turns"}:
            source_ids.update(map(str, record.get("source_utterance_ids", [])))
    return plan_ids, source_ids


def _cached_episode(path: Path, recipe_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = read_json(path)
    if record.get("recipe_sha256") != recipe_hash:
        return None
    for field in ("mic_audio", "target_speech"):
        asset = Path(record[field])
        if not asset.is_file() or sha256_file(asset) != record.get(f"{field}_sha256"):
            return None
    if record.get("screens"):
        screen = Path(record["screens"])
        if not screen.is_file() or sha256_file(screen) != record.get("screens_sha256"):
            return None
    return record


def _screens(
    root: Path,
    dataset: str,
    episode_id: str,
    ticks: int,
    fixture: bool,
    screen_command: str | None,
) -> str:
    output = root / "normalized" / "screens" / dataset / f"{episode_id}.npz"
    receipt_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = {
        "episode_id": episode_id,
        "ticks": ticks,
        "tick_ms": 80,
        "height": 224,
        "width": 224,
        "fixture": fixture,
        "adapter": screen_command,
    }
    recipe_hash = stable_hash(recipe)
    if output.exists() and receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("recipe_sha256") == recipe_hash and receipt.get(
            "artifact_sha256"
        ) == sha256_file(output):
            return relative_to_root(output, root)
    if fixture:
        frame = np.zeros((1, 3, 224, 224), dtype=np.float32)
        frame[:, 0, 48:176, 48:176] = 0.25
        np.savez_compressed(output, ticks=np.asarray([min(1, ticks - 1)]), frames=frame)
    else:
        if not screen_command:
            raise ValueError("screen-conditioned episodes require --screen-command")
        request = {"operation": "capture-isolated-sandbox", **recipe, "output": str(output)}
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(request, handle)
            handle.flush()
            result = subprocess.run(
                [
                    *shlex.split(screen_command),
                    "--request",
                    handle.name,
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode or not output.is_file():
            raise RuntimeError(f"screen adapter failed: {(result.stderr or result.stdout)[-1000:]}")
    write_json(
        receipt_path,
        {"recipe_sha256": recipe_hash, "artifact_sha256": sha256_file(output)},
    )
    return relative_to_root(output, root)


def _compose_plan(
    root: Path,
    dataset: str,
    plan: dict[str, Any],
    receipts: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, Any],
    fixture: bool,
    screen_command: str | None,
) -> dict[str, Any]:
    episode_id = f"{dataset}-plan-{plan['plan_id']}"
    turn_receipts = []
    for turn in plan["turns"]:
        key = (str(plan["plan_id"]), str(turn["turn_id"]))
        if key not in receipts:
            raise ValueError(f"synthesis receipt is absent for {key}")
        turn_receipts.append(receipts[key])
    recipe_hash = stable_hash(
        {
            "plan": plan["recipe_sha256"],
            "receipts": [
                {
                    "recipe_sha256": receipt["recipe_sha256"],
                    "audio_sha256": receipt["audio_sha256"],
                }
                for receipt in turn_receipts
            ],
            "voice_registry": registry["registry_sha256"],
            "screen_adapter": screen_command if plan["category"] == "screen_task" else None,
            "fixture": fixture,
        }
    )
    audio_dir = root / "normalized" / "episodes" / dataset
    record_path = audio_dir / f"{episode_id}.json"
    cached = _cached_episode(record_path, recipe_hash)
    if cached is not None:
        return cached
    cursor = 2 * FRAME_SAMPLES
    mic = np.zeros(
        align_up(int(float(plan["target_duration_seconds"]) * SAMPLE_RATE)), dtype=np.float32
    )
    target = np.zeros_like(mic)
    turns = []
    segments = []
    user_voice_id = ""
    for turn in plan["turns"]:
        key = (str(plan["plan_id"]), str(turn["turn_id"]))
        receipt = receipts[key]
        waveform = read_mono(root / receipt["audio"])
        if turn["role"] == "assistant":
            cursor = align_up(cursor + 3 * FRAME_SAMPLES)
            end = cursor + waveform.size
            needed = align_up(end) + FRAME_SAMPLES
            if needed > target.size:
                extension = needed - target.size
                mic = np.pad(mic, (0, extension))
                target = np.pad(target, (0, extension))
            target[cursor:end] = waveform
            segments.append({"turn_id": turn["turn_id"], "start_sample": cursor, "end_sample": end})
        else:
            cursor = align_up(cursor)
            end = cursor + waveform.size
            if end > mic.size:
                extension = align_up(end) + FRAME_SAMPLES - mic.size
                mic = np.pad(mic, (0, extension))
                target = np.pad(target, (0, extension))
            mic[cursor:end] = waveform
            user_voice_id = str(receipt["voice_id"])
        turns.append(
            {
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
                "start_sample": cursor,
                "end_sample": end,
            }
        )
        cursor = end
    timeline = max(mic.size, align_up(cursor) + 3 * FRAME_SAMPLES)
    mic = np.pad(mic, (0, timeline - mic.size))
    target = np.pad(target, (0, timeline - target.size))
    mic_path = audio_dir / f"{episode_id}-mic.flac"
    target_path = audio_dir / f"{episode_id}-target.flac"
    write_flac(mic_path, mic)
    write_flac(target_path, target)
    source_license = "internal-generated-plans; CosyVoice example voices"
    license_hash = sha256_file(root / "voices" / "registry.json")
    record = {
        "episode_id": episode_id,
        "plan_id": plan["plan_id"],
        "mic_audio": str(mic_path.resolve()),
        "mic_audio_sha256": sha256_file(mic_path),
        "target_speech": str(target_path.resolve()),
        "target_speech_sha256": sha256_file(target_path),
        "source": "e2-reviewed-computer-dialogue",
        "source_version": "pilot-plan-v1",
        "source_url": "internal://e2-pilot/text-plans",
        "source_utterance_ids": [plan["plan_id"]],
        "source_license": source_license,
        "redistribution_allowed": False,
        "license_sha256": license_hash,
        "template_id": plan["template_id"],
        "intent": plan["intent"],
        "language": plan["language"],
        "split": plan["split"],
        "user_voice_id": user_voice_id,
        "assistant_voice_id": registry["assistant_voice_id"],
        "session_id_hash": stable_hash({"scenario": plan["scenario_id"]}),
        "device_id_hash": "isolated-sandbox" if plan["category"] == "screen_task" else "synthetic",
        "scenario": plan["scenario_id"],
        "category": plan["category"],
        "turns": turns,
        "target_segments": segments,
        "recipe_sha256": recipe_hash,
        "fixture": fixture,
    }
    if plan["category"] == "screen_task":
        screen_path = _screens(
            root, dataset, episode_id, timeline // FRAME_SAMPLES, fixture, screen_command
        )
        record["screens"] = str((root / screen_path).resolve())
        record["screens_sha256"] = sha256_file(root / screen_path)
    write_json(record_path, record)
    return record


def _compose_source(
    root: Path,
    dataset: str,
    item: dict[str, Any],
    receipts: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, Any],
    fixture: bool,
) -> dict[str, Any]:
    episode_id = f"{dataset}-source-{item['source_item_id']}"
    response = receipts.get((str(item["source_item_id"]), "assistant-response"))
    recipe_hash = stable_hash(
        {
            "source_audio": item["audio_sha256"],
            "source_item": item["source_item_id"],
            "response": response.get("recipe_sha256") if response else None,
            "response_audio": response.get("audio_sha256") if response else None,
            "voice_registry": registry["registry_sha256"],
            "dataset": dataset,
            "fixture": fixture,
        }
    )
    audio_dir = root / "normalized" / "episodes" / dataset
    record_path = audio_dir / f"{episode_id}.json"
    cached = _cached_episode(record_path, recipe_hash)
    if cached is not None:
        return cached
    source_audio = read_mono(root / item["audio"])
    lead = 2 * FRAME_SAMPLES
    mic_size = align_up(lead + source_audio.size) + 4 * FRAME_SAMPLES
    mic = np.zeros(mic_size, dtype=np.float32)
    target = np.zeros(mic_size, dtype=np.float32)
    mic[lead : lead + source_audio.size] = source_audio
    turns = [
        {
            "turn_id": "source-input",
            "role": "user",
            "text": item.get("text", ""),
            "start_sample": lead,
            "end_sample": lead + source_audio.size,
        }
    ]
    segments: list[dict[str, Any]] = []
    if item["category"] == "adjacent_turns":
        receipt = response
        if receipt is None:
            raise ValueError(f"adjacent response receipt is absent for {item['source_item_id']}")
        waveform = read_mono(root / receipt["audio"])
        start = align_up(lead + source_audio.size + 4 * FRAME_SAMPLES)
        end = start + waveform.size
        timeline = align_up(end) + 4 * FRAME_SAMPLES
        mic = np.pad(mic, (0, max(0, timeline - mic.size)))
        target = np.pad(target, (0, max(0, timeline - target.size)))
        target[start:end] = waveform
        turns.append(
            {
                "turn_id": "assistant-response",
                "role": "assistant",
                "text": item["response_text"],
                "start_sample": start,
                "end_sample": end,
            }
        )
        segments.append({"turn_id": "assistant-response", "start_sample": start, "end_sample": end})
    mic_path = audio_dir / f"{episode_id}-mic.flac"
    target_path = audio_dir / f"{episode_id}-target.flac"
    write_flac(mic_path, mic)
    write_flac(target_path, target)
    record = {
        "episode_id": episode_id,
        "mic_audio": str(mic_path.resolve()),
        "mic_audio_sha256": sha256_file(mic_path),
        "target_speech": str(target_path.resolve()),
        "target_speech_sha256": sha256_file(target_path),
        "source": item["source_id"],
        "source_version": item["source_version"],
        "source_url": item["source_url"],
        "source_utterance_ids": item["source_utterance_ids"],
        "source_license": item["source_license"],
        "redistribution_allowed": item["redistribution_allowed"],
        "license_sha256": item["license_sha256"],
        "template_id": (
            f"source-{'adjacent' if segments else 'input'}-"
            f"{item['source_id']}-{item['language']}-{item['split']}-v1"
        ),
        "intent": "conversational-response" if segments else "listen-observe",
        "language": item["language"],
        "split": item["split"],
        "user_voice_id": item.get("speaker_id", "unknown-source-speaker"),
        "assistant_voice_id": registry["assistant_voice_id"],
        "session_id_hash": stable_hash(
            {"source": item["source_id"], "session": item["session_id"]}
        ),
        "device_id_hash": str(item["source_id"]),
        "scenario": str(item["source_item_id"]),
        "category": item["category"],
        "turns": turns,
        "target_segments": segments,
        "recipe_sha256": recipe_hash,
        "source_normalization": item.get("normalization"),
        "fixture": fixture,
    }
    write_json(record_path, record)
    return record


def _verify_isolation(records: list[dict[str, Any]]) -> None:
    dimensions = {
        "speaker": "user_voice_id",
        "session": "session_id_hash",
        "template": "template_id",
        "scenario": "scenario",
    }
    for label, field in dimensions.items():
        seen: dict[str, set[str]] = defaultdict(set)
        for record in records:
            seen[str(record[field])].add(str(record["split"]))
        leaks = [value for value, splits in seen.items() if len(splits) > 1]
        if leaks:
            raise ValueError(f"{label} values cross dataset splits: {leaks[:3]}")


def _duration_seconds(root: Path, record: dict[str, Any]) -> float:
    path = Path(record["mic_audio"])
    if not path.is_absolute():
        path = root / path
    return read_mono(path).size / SAMPLE_RATE


def _select_sources(
    root: Path,
    dataset: str,
    source_items: list[dict[str, Any]],
    receipts: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, Any],
    excluded_sources: set[str],
    fixture: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in source_items:
        source_ids = set(map(str, item.get("source_utterance_ids", [])))
        if source_ids & excluded_sources:
            continue
        grouped[(item["category"], item["language"], item["split"])].append(item)
    spec = dataset_spec(dataset)
    for key, items in grouped.items():
        items.sort(key=lambda value: value["source_item_id"])
        if fixture:
            candidates = items[::2] if dataset == "canary" else items
        else:
            candidates = items
        target = spec.target_seconds(*key)
        elapsed = 0.0
        for item in candidates:
            record = _compose_source(root, dataset, item, receipts, registry, fixture)
            duration = _duration_seconds(root, record)
            if not fixture and elapsed and elapsed + duration > target * 1.02:
                continue
            selected.append(record)
            elapsed += duration
            if not fixture and elapsed >= target * 0.998:
                break
        if not fixture and elapsed < target * 0.98:
            raise ValueError(
                f"not enough independent source audio for {key}: {elapsed:.1f}s < {target:.1f}s"
            )
    return selected


def build_pilot_manifest(
    root: str | Path,
    *,
    dataset: str,
    fixture: bool = False,
    normalize_command: str | None = None,
    screen_command: str | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    if normalize_command:
        _run_inventory_adapter(normalize_command, root)
    elif not fixture:
        inventory = root / "normalized" / "source-items.jsonl"
        receipt = root / "normalized" / "source-items.receipt.json"
        if not inventory.is_file() or not receipt.is_file():
            raise ValueError(
                "production manifest construction requires --normalize-command on first run"
            )
    source_items, plans, receipts, registry = _load_assets(root, dataset)
    excluded_plans, excluded_sources = _excluded(root, dataset, fixture)
    records = _select_sources(
        root,
        dataset,
        source_items,
        receipts,
        registry,
        excluded_sources,
        fixture,
    )
    for plan in plans["plans"]:
        if plan["plan_id"] in excluded_plans:
            continue
        records.append(
            _compose_plan(root, dataset, plan, receipts, registry, fixture, screen_command)
        )
    ids = [record["episode_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("episode IDs are duplicated")
    _verify_isolation(records)
    destination = root / "manifests" / dataset
    for split in SPLITS:
        write_jsonl(destination / f"{split}.jsonl", [r for r in records if r["split"] == split])
    manifest_path = destination / "episodes.jsonl"
    write_jsonl(manifest_path, sorted(records, key=lambda value: value["episode_id"]))
    quota = defaultdict(float)
    for record in records:
        samples = read_mono(Path(record["mic_audio"])).size
        quota[(record["category"], record["language"], record["split"])] += samples / SAMPLE_RATE
    report = {
        "dataset": dataset,
        "fixture": fixture,
        "episodes": len(records),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "canary_excluded_plan_ids": len(excluded_plans),
        "canary_excluded_source_ids": len(excluded_sources),
        "quotas": [
            {
                "category": category,
                "language": language,
                "split": split,
                "actual_seconds": quota[(category, language, split)],
                "target_seconds": dataset_spec(dataset).target_seconds(category, language, split),
            }
            for category in CATEGORIES
            for language in LANGUAGES
            for split in SPLITS
        ],
    }
    write_json(root / "reports" / f"{dataset}-manifest-report.json", report)
    return report
