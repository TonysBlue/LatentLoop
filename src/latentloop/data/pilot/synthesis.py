from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from latentloop.data.pilot.audio import fixture_voice, quality_metrics, write_flac
from latentloop.data.pilot.common import (
    ensure_tree,
    read_json,
    relative_to_root,
    require_sha256,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from latentloop.data.pilot.text import plan_recipe_sha256


def _run_adapter(command: str, request: dict[str, Any], output: Path) -> None:
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError("adapter command is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(request, request_file, ensure_ascii=False)
        request_file.flush()
        completed = subprocess.run(
            [*arguments, "--request", request_file.name, "--output", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"adapter failed ({completed.returncode}): {detail[-1000:]}")
    if not output.is_file():
        raise RuntimeError("adapter completed without creating its requested output")


def _asr_score(
    command: str | None, path: Path, text: str, language: str, fixture: bool
) -> tuple[str, float]:
    metric = "cer" if language == "zh" else "wer"
    if fixture:
        return metric, 0.0
    if not command:
        raise ValueError("production synthesis requires --asr-command for CER/WER gating")
    with tempfile.TemporaryDirectory() as temporary:
        result_path = Path(temporary) / "asr.json"
        _run_adapter(
            command,
            {"operation": "transcribe", "audio": str(path), "text": text, "language": language},
            result_path,
        )
        result = read_json(result_path)
    if result.get("metric") != metric:
        raise ValueError(f"ASR adapter must return the {metric} metric for {language}")
    score = float(result["score"])
    if not 0 <= score <= 1:
        raise ValueError("ASR score must be in [0, 1]")
    return metric, score


def _production_audio_report(path: Path) -> dict[str, Any]:
    report_path = path.with_suffix(".metrics.json")
    if not report_path.is_file():
        raise ValueError(
            "production TTS adapter must write a sibling .metrics.json with integrated_lufs"
        )
    report = read_json(report_path)
    loudness = float(report["integrated_lufs"])
    if not -24.0 <= loudness <= -22.0:
        raise ValueError(f"synthesized loudness {loudness:.2f} LUFS is outside -23 +/- 1")
    return {"integrated_lufs": loudness, "metrics_sha256": sha256_file(report_path)}


def synthesize_pilot(
    root: str | Path,
    *,
    dataset: str,
    fixture: bool = False,
    synth_command: str | None = None,
    asr_command: str | None = None,
    model_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    plans_path = root / "text" / f"{dataset}-plans.json"
    registry_path = root / "voices" / "registry.json"
    plans = read_json(plans_path)
    registry = read_json(registry_path)
    if plans.get("dataset") != dataset:
        raise ValueError("text plan dataset does not match the requested dataset")
    stale = [
        plan["plan_id"]
        for plan in plans["plans"]
        if plan.get("recipe_sha256") != plan_recipe_sha256(plan)
    ]
    if stale:
        raise ValueError(f"text plan recipe hashes are stale: {stale[:3]}")
    if fixture:
        model_hash = "fixture"
    else:
        model_hash = require_sha256(model_sha256, "TTS model")
        if not synth_command:
            raise ValueError("production synthesis requires --synth-command")
    voices = {voice["voice_id"]: voice for voice in registry["voices"]}
    assistant_voice = str(registry["assistant_voice_id"])
    user_voices = {
        (language, split): sorted(
            voice["voice_id"]
            for voice in voices.values()
            if voice.get("role") == "user"
            and voice.get("language") in {language, "multilingual"}
            and voice.get("split") == split
        )
        for language in ("zh", "en")
        for split in ("train", "validation", "test")
    }
    receipts: list[dict[str, Any]] = []
    rejected = 0
    for plan_index, plan in enumerate(plans["plans"]):
        language = str(plan["language"])
        split = str(plan["split"])
        available_voices = user_voices[(language, split)]
        if not available_voices:
            raise ValueError(f"no {language} user voice is assigned to split {split}")
        user_voice = available_voices[plan_index % len(available_voices)]
        for turn in plan["turns"]:
            role = str(turn["role"])
            voice_id = assistant_voice if role == "assistant" else user_voice
            prompt_hash = str(voices[voice_id]["prompt_sha256"])
            recipe = {
                "plan_id": plan["plan_id"],
                "plan_recipe_sha256": plan["recipe_sha256"],
                "turn_id": turn["turn_id"],
                "text": turn["text"],
                "language": language,
                "role": role,
                "voice_id": voice_id,
                "voice_prompt_sha256": prompt_hash,
                "model_sha256": model_hash,
            }
            recipe_hash = stable_hash(recipe)
            output = root / "synthesized" / dataset / plan["plan_id"] / f"{turn['turn_id']}.flac"
            receipt_path = output.with_suffix(".json")
            if output.is_file() and receipt_path.is_file():
                old = read_json(receipt_path)
                if old.get("recipe_sha256") == recipe_hash and old.get(
                    "audio_sha256"
                ) == sha256_file(output):
                    receipts.append(old)
                    continue
            attempts = 0
            while True:
                attempts += 1
                if fixture:
                    write_flac(output, fixture_voice(turn["text"], plan_index + attempts))
                else:
                    _run_adapter(synth_command or "", {**recipe, "attempt": attempts}, output)
                metrics = quality_metrics(output)
                if metrics["duration_seconds"] < 0.3 or metrics["duration_seconds"] > 15:
                    raise ValueError(f"synthesized utterance duration is invalid: {output}")
                if not metrics["finite"] or metrics["clipping_fraction"] > 0.001:
                    raise ValueError(f"synthesized utterance quality is invalid: {output}")
                metric, score = _asr_score(asr_command, output, turn["text"], language, fixture)
                if score <= 0.20 or attempts == 2:
                    break
            if score > 0.20:
                output.unlink(missing_ok=True)
                rejected += 1
                continue
            receipt = {
                **recipe,
                "recipe_sha256": recipe_hash,
                "audio": relative_to_root(output, root),
                "audio_sha256": sha256_file(output),
                "duration_seconds": metrics["duration_seconds"],
                "asr_metric": metric,
                "asr_score": score,
                "attempts": attempts,
                "fixture": fixture,
            }
            if not fixture:
                receipt["normalization"] = _production_audio_report(output)
            write_json(receipt_path, receipt)
            receipts.append(receipt)
    source_inventory = root / "normalized" / "source-items.jsonl"
    if source_inventory.exists():
        from latentloop.data.pilot.common import read_jsonl

        for source_index, item in enumerate(read_jsonl(source_inventory)):
            if item.get("category") != "adjacent_turns" or not item.get("response_text"):
                continue
            language = str(item["language"])
            text = str(item["response_text"])
            turn_id = "assistant-response"
            recipe = {
                "source_item_id": item["source_item_id"],
                "turn_id": turn_id,
                "text": text,
                "language": language,
                "role": "assistant",
                "voice_id": assistant_voice,
                "voice_prompt_sha256": voices[assistant_voice]["prompt_sha256"],
                "model_sha256": model_hash,
            }
            recipe_hash = stable_hash(recipe)
            output = (
                root
                / "synthesized"
                / dataset
                / "source-responses"
                / f"{item['source_item_id']}.flac"
            )
            receipt_path = output.with_suffix(".json")
            if output.is_file() and receipt_path.is_file():
                old = read_json(receipt_path)
                if old.get("recipe_sha256") == recipe_hash and old.get(
                    "audio_sha256"
                ) == sha256_file(output):
                    receipts.append(old)
                    continue
            attempts = 0
            while True:
                attempts += 1
                if fixture:
                    write_flac(output, fixture_voice(text, source_index + attempts))
                else:
                    _run_adapter(synth_command or "", {**recipe, "attempt": attempts}, output)
                metrics = quality_metrics(output)
                if metrics["duration_seconds"] < 0.3 or metrics["duration_seconds"] > 15:
                    raise ValueError(f"synthesized utterance duration is invalid: {output}")
                if not metrics["finite"] or metrics["clipping_fraction"] > 0.001:
                    raise ValueError(f"synthesized utterance quality is invalid: {output}")
                metric, score = _asr_score(asr_command, output, text, language, fixture)
                if score <= 0.20 or attempts == 2:
                    break
            if score > 0.20:
                output.unlink(missing_ok=True)
                rejected += 1
                continue
            receipt = {
                **recipe,
                "recipe_sha256": recipe_hash,
                "audio": relative_to_root(output, root),
                "audio_sha256": sha256_file(output),
                "duration_seconds": metrics["duration_seconds"],
                "asr_metric": metric,
                "asr_score": score,
                "attempts": attempts,
                "fixture": fixture,
            }
            if not fixture:
                receipt["normalization"] = _production_audio_report(output)
            write_json(receipt_path, receipt)
            receipts.append(receipt)
    path = root / "synthesized" / f"{dataset}-utterances.jsonl"
    write_jsonl(path, receipts)
    report = {
        "dataset": dataset,
        "fixture": fixture,
        "utterances": len(receipts),
        "rejected": rejected,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "tts_model_sha256": model_hash,
        "prompt_registry_sha256": sha256_file(registry_path),
    }
    write_json(root / "reports" / f"{dataset}-synthesis-report.json", report)
    return report
