from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentloop.config import DataConfig, ModelConfig
from latentloop.data import import_speech_manifest
from latentloop.data.pilot import (
    audit_pilot_data,
    build_pilot_manifest,
    build_pilot_text,
    fetch_pilot_data,
    select_pilot_voices,
    synthesize_pilot,
)
from latentloop.data.pilot.common import read_jsonl, sha256_file
from latentloop.types import SpeechControl


def _fixture_pipeline(root: Path, dataset: str) -> None:
    build_pilot_text(root, dataset=dataset, fixture=True)
    synthesize_pilot(root, dataset=dataset, fixture=True)
    build_pilot_manifest(root, dataset=dataset, fixture=True)
    audit_pilot_data(root, dataset=dataset, fixture=True)


def test_fixture_pipeline_is_resumable_and_isolates_canary(tmp_path: Path) -> None:
    root = tmp_path / "e2-pilot"
    first = fetch_pilot_data(root, fixture=True)
    select_pilot_voices(root, fixture=True)
    _fixture_pipeline(root, "canary")
    _fixture_pipeline(root, "pilot")

    canary = read_jsonl(root / "manifests" / "canary" / "episodes.jsonl")
    pilot = read_jsonl(root / "manifests" / "pilot" / "episodes.jsonl")
    canary_sources = {source for record in canary for source in record["source_utterance_ids"]}
    pilot_sources = {source for record in pilot for source in record["source_utterance_ids"]}
    assert canary_sources.isdisjoint(pilot_sources)
    assert {record["plan_id"] for record in canary if record.get("plan_id")}.isdisjoint(
        {record["plan_id"] for record in pilot if record.get("plan_id")}
    )
    manifest = root / "manifests" / "pilot" / "episodes.jsonl"
    before = sha256_file(manifest)

    second = fetch_pilot_data(root, fixture=True)
    _fixture_pipeline(root, "pilot")

    assert first["inventory_sha256"] == second["inventory_sha256"]
    assert sha256_file(manifest) == before
    assert (root / "reports" / "pilot" / "data-card.md").is_file()
    assert (root / "reports" / "pilot" / "license-report.json").is_file()
    assert (root / "reports" / "pilot" / "quota-report.json").is_file()
    assert (root / "reports" / "pilot" / "quality-report.json").is_file()


def test_pilot_manifest_import_has_segment_aware_training_masks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "e2-pilot"
    fetch_pilot_data(root, fixture=True)
    select_pilot_voices(root, fixture=True)
    _fixture_pipeline(root, "canary")
    records = read_jsonl(root / "manifests" / "canary" / "episodes.jsonl")
    public_record = next(record for record in records if record["category"] == "public_speech")
    dialogue_record = next(
        record for record in records if record["category"] == "synthetic_dialogue"
    )
    manifest = root / "single.jsonl"

    manifest.write_text(json.dumps(public_record) + "\n", encoding="utf-8")
    data = DataConfig()
    model = ModelConfig()
    public_episode = next(import_speech_manifest(manifest, data, model))
    assert not any(bool(unit.speech_mask.item()) for unit in public_episode.units)
    assert all(
        unit.control_target.speech.item() == SpeechControl.SILENT for unit in public_episode.units
    )

    manifest.write_text(json.dumps(dialogue_record) + "\n", encoding="utf-8")
    dialogue = next(import_speech_manifest(manifest, data, model))
    controls = [unit.control_target.speech.item() for unit in dialogue.units]
    masks = [bool(unit.speech_mask.item()) for unit in dialogue.units]
    assert SpeechControl.START in controls
    assert SpeechControl.STOP in controls
    assert all(
        mask == (control in {SpeechControl.START, SpeechControl.CONTINUE, SpeechControl.STOP})
        for control, mask in zip(controls, masks, strict=True)
    )


def test_production_fetch_requires_reviewed_source_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --lock"):
        fetch_pilot_data(tmp_path / "e2-pilot")
    assert (tmp_path / "e2-pilot" / "raw" / "source-lock.template.json").is_file()


def test_production_text_plan_meets_scale_and_duration_mix(tmp_path: Path) -> None:
    root = tmp_path / "e2-pilot"
    report = build_pilot_text(root, dataset="pilot")
    plans = json.loads(Path(report["path"]).read_text(encoding="utf-8"))["plans"]

    assert len(plans) == 1_200
    assert report["assistant_responses"] >= 2_000
    assert sum(plan["language"] == "zh" for plan in plans) == 960
    assert sum(plan["language"] == "en" for plan in plans) == 240
    assert sum(plan["duration_class"] == "short" for plan in plans) == 720
    assert sum(plan["duration_class"] == "medium" for plan in plans) == 300
    assert sum(plan["duration_class"] == "long" for plan in plans) == 180
    assert sum(len(plan["turns"]) > 2 for plan in plans) == 480
    assert sum(
        plan["target_duration_seconds"]
        for plan in plans
        if plan["category"] == "synthetic_dialogue"
    ) == pytest.approx(18_000, abs=0.001)
    assert sum(
        plan["target_duration_seconds"] for plan in plans if plan["category"] == "screen_task"
    ) == pytest.approx(1_800, abs=0.001)
    assert report["pending_review"] == 1_200
