from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from latentloop.data.pilot.audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    quality_metrics,
    read_mono,
)
from latentloop.data.pilot.common import (
    CATEGORIES,
    LANGUAGES,
    SPLITS,
    ensure_tree,
    read_json,
    read_jsonl,
    require_sha256,
    sha256_file,
    stable_hash,
    write_json,
)
from latentloop.data.pilot.spec import (
    CATEGORY_FRACTIONS,
    LANGUAGE_FRACTIONS,
    MIMI_CODEC_ID,
    MIMI_REVISION,
    MIMI_WEIGHT_SHA256,
    SPLIT_FRACTIONS,
    dataset_spec,
)

REQUIRED_FIELDS = (
    "episode_id",
    "mic_audio",
    "mic_audio_sha256",
    "target_speech",
    "target_speech_sha256",
    "source",
    "source_version",
    "source_url",
    "source_utterance_ids",
    "source_license",
    "redistribution_allowed",
    "license_sha256",
    "template_id",
    "intent",
    "language",
    "split",
    "user_voice_id",
    "assistant_voice_id",
    "turns",
    "target_segments",
    "recipe_sha256",
)


def _fraction_error(actual: float, target: float) -> float:
    return abs(actual - target) / target if target else float(actual != 0)


def _validate_record(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{record.get('episode_id', '<unknown>')} is missing {missing}")
    if record["category"] not in CATEGORIES:
        raise ValueError(f"unknown category: {record['category']}")
    if record["language"] not in LANGUAGES or record["split"] not in SPLITS:
        raise ValueError("language or split is outside the locked Pilot vocabulary")
    require_sha256(record["recipe_sha256"], "recipe")
    require_sha256(record["license_sha256"], "license")
    metrics: dict[str, Any] = {}
    for field in ("mic_audio", "target_speech"):
        path = Path(record[field])
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"audio asset is absent: {path}")
        expected = require_sha256(record[f"{field}_sha256"], field)
        if sha256_file(path) != expected:
            raise ValueError(f"audio hash mismatch: {path}")
        metrics[field] = quality_metrics(path)
        if not metrics[field]["finite"] or metrics[field]["clipping_fraction"] > 0.001:
            raise ValueError(f"audio quality gate failed: {path}")
        if metrics[field]["peak_dbfs"] > -1.0:
            raise ValueError(f"audio peak exceeds -1 dBFS: {path}")
    if metrics["mic_audio"]["samples"] != metrics["target_speech"]["samples"]:
        raise ValueError("mic and target audio timelines differ")
    timeline_samples = int(metrics["mic_audio"]["samples"])
    if timeline_samples % FRAME_SAMPLES:
        raise ValueError("episode timeline is not aligned to the 80 ms clock")
    previous_stop = -1
    starts = 0
    stops = 0
    turns = {str(turn["turn_id"]): turn for turn in record["turns"]}
    if len(turns) != len(record["turns"]):
        raise ValueError("turn IDs are duplicated within an episode")
    for turn in record["turns"]:
        start = int(turn["start_sample"])
        end = int(turn["end_sample"])
        duration = end - start
        if start < 0 or end > timeline_samples or duration < int(0.3 * SAMPLE_RATE):
            raise ValueError("turn contains fewer than 300 ms of speech or is out of bounds")
        if duration > 15 * SAMPLE_RATE:
            raise ValueError("turn contains more than 15 seconds of speech")
    for segment in record["target_segments"]:
        start = int(segment["start_sample"])
        end = int(segment["end_sample"])
        if start % FRAME_SAMPLES:
            raise ValueError("assistant START is not on an 80 ms boundary")
        final_frame = (end - 1) // FRAME_SAMPLES
        stop_frame = final_frame + 1
        if start < 0 or end <= start or stop_frame * FRAME_SAMPLES >= timeline_samples:
            raise ValueError("assistant segment is outside the episode timeline")
        if start // FRAME_SAMPLES <= previous_stop:
            raise ValueError("assistant segments overlap")
        turn = turns.get(str(segment["turn_id"]))
        if turn is None or turn["role"] != "assistant":
            raise ValueError("target segment does not reference an assistant turn")
        if (
            abs(int(turn["start_sample"]) - start) > FRAME_SAMPLES
            or abs(int(turn["end_sample"]) - end) > FRAME_SAMPLES
        ):
            raise ValueError("target boundary differs from its turn by more than one tick")
        previous_stop = stop_frame
        starts += 1
        stops += 1
    if record["category"] == "public_speech" and record["target_segments"]:
        raise ValueError("public real-input-only episodes must not have speech targets")
    if record["category"] == "public_speech":
        target_path = Path(record["target_speech"])
        if not target_path.is_absolute():
            target_path = root / target_path
        if float(abs(read_mono(target_path)).max(initial=0.0)) > 0:
            raise ValueError("public real-input-only episode target must be silent")
    if record["category"] in {"public_speech", "adjacent_turns"} and not record.get(
        "fixture", False
    ):
        normalization = record.get("source_normalization", {})
        loudness = float(normalization["integrated_lufs"])
        if not -24.0 <= loudness <= -22.0:
            raise ValueError("source audio is outside -23 +/- 1 LUFS")
        require_sha256(normalization.get("metrics_sha256"), "source loudness metrics")
    if record.get("screens"):
        screen_path = Path(record["screens"])
        if not screen_path.is_absolute():
            screen_path = root / screen_path
        if not screen_path.is_file():
            raise FileNotFoundError(f"screen artifact is absent: {screen_path}")
        expected_screen = require_sha256(record.get("screens_sha256"), "screen artifact")
        if sha256_file(screen_path) != expected_screen:
            raise ValueError("screen artifact hash mismatch")
    return {
        "episode_id": record["episode_id"],
        "duration_seconds": timeline_samples / SAMPLE_RATE,
        "starts": starts,
        "stops": stops,
        "audio": metrics,
    }


def _leakage(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    fields = {
        "speaker": "user_voice_id",
        "session": "session_id_hash",
        "template": "template_id",
        "scenario": "scenario",
    }
    result: dict[str, list[str]] = {}
    for label, field in fields.items():
        groups: dict[str, set[str]] = defaultdict(set)
        for record in records:
            groups[str(record[field])].add(str(record["split"]))
        result[label] = sorted(group for group, splits in groups.items() if len(splits) > 1)
    return result


def _review_report(
    root: Path,
    dataset: str,
    records: list[dict[str, Any]],
    duration: dict[str, float],
    fixture: bool,
) -> dict[str, Any]:
    path = root / "reports" / "reviews" / f"{dataset}.jsonl"
    if fixture:
        return {"required": False, "reviewed_seconds": 0.0, "path": str(path)}
    if not path.is_file():
        raise ValueError(f"manual review ledger is required: {path}")
    reviews = read_jsonl(path)
    review_ids = [str(review["episode_id"]) for review in reviews]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("manual review ledger contains duplicate episode IDs")
    by_id = {record["episode_id"]: record for record in records}
    reviewed_seconds = 0.0
    severe = 0
    minor = 0
    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for review in reviews:
        episode_id = str(review["episode_id"])
        if episode_id not in by_id:
            raise ValueError(f"review references an unknown episode: {episode_id}")
        reviewed = float(review["reviewed_seconds"])
        if reviewed <= 0:
            raise ValueError("reviewed_seconds must be positive")
        reviewed_seconds += min(reviewed, duration[episode_id])
        severity = str(review.get("severity", "none"))
        severe += severity == "severe"
        minor += severity == "minor"
        record = by_id[episode_id]
        sources[str(record["source"])] += 1
        languages[str(record["language"])] += 1
        splits[str(record["split"])] += 1
    total = sum(duration.values())
    required_fraction = 0.10 if dataset == "canary" else 0.02
    if reviewed_seconds < total * required_fraction:
        raise ValueError("manual review duration is below the required fraction")
    if severe:
        raise ValueError("manual review contains severe errors")
    minor_limit = 0.02 if dataset == "canary" else 0.01
    if reviews and minor / len(reviews) >= minor_limit:
        raise ValueError("manual review minor-error rate exceeds the gate")
    if dataset == "canary":
        under_reviewed = [
            f"source={value}"
            for value in {str(record["source"]) for record in records}
            if sources[value] < 20
        ]
        under_reviewed.extend(
            f"language={value}"
            for value in {str(record["language"]) for record in records}
            if languages[value] < 20
        )
        under_reviewed.extend(
            f"split={value}"
            for value in {str(record["split"]) for record in records}
            if splits[value] < 20
        )
        if under_reviewed:
            raise ValueError(
                f"Canary review dimensions have fewer than 20 items: {under_reviewed[:3]}"
            )
    return {
        "required": True,
        "items": len(reviews),
        "reviewed_seconds": reviewed_seconds,
        "reviewed_fraction": reviewed_seconds / total,
        "severe_errors": severe,
        "minor_errors": minor,
        "path": str(path),
    }


def _mimi_report(
    root: Path, dataset: str, fixture: bool, path: str | Path | None
) -> dict[str, Any]:
    report_path = (
        Path(path).expanduser() if path else root / "reports" / f"{dataset}-mimi-decode.json"
    )
    if fixture and not report_path.exists():
        return {"required": False, "checked_segments": 0, "path": str(report_path)}
    if not report_path.is_file():
        raise ValueError(f"Mimi decode-check report is required: {report_path}")
    report = read_json(report_path)
    if int(report.get("checked_segments", 0)) < 100 or int(report.get("failed_segments", 0)):
        raise ValueError("Mimi decode-check requires at least 100 successful target segments")
    weight_hash = require_sha256(report.get("mimi_weight_sha256"), "Mimi weight")
    if weight_hash != MIMI_WEIGHT_SHA256:
        raise ValueError("Mimi decode-check used an unexpected weight identity")
    if report.get("codec_id") != MIMI_CODEC_ID or report.get("codec_revision") != MIMI_REVISION:
        raise ValueError("Mimi decode-check codec identity does not match E2")
    return report


def _check_canary_exclusion(root: Path, records: list[dict[str, Any]]) -> None:
    path = root / "manifests" / "canary" / "episodes.jsonl"
    if not path.is_file():
        raise ValueError("Pilot audit requires an audited Canary manifest")
    audit_path = root / "reports" / "canary-audit.json"
    if not audit_path.is_file():
        raise ValueError("Pilot audit requires a Canary audit report")
    audit = read_json(audit_path)
    if not audit.get("passed") or audit.get("manifest_sha256") != sha256_file(path):
        raise ValueError("Canary manifest is not covered by its audit report")
    canary = read_jsonl(path)
    canary_plans = {str(record.get("plan_id")) for record in canary if record.get("plan_id")}
    pilot_plans = {str(record.get("plan_id")) for record in records if record.get("plan_id")}
    canary_sources = {
        str(item)
        for record in canary
        if record.get("category") in {"public_speech", "adjacent_turns"}
        for item in record.get("source_utterance_ids", [])
    }
    pilot_sources = {
        str(item)
        for record in records
        if record.get("category") in {"public_speech", "adjacent_turns"}
        for item in record.get("source_utterance_ids", [])
    }
    if canary_plans & pilot_plans or canary_sources & pilot_sources:
        raise ValueError("Pilot reuses a Canary plan or source utterance")


def audit_pilot_data(
    root: str | Path,
    *,
    dataset: str,
    fixture: bool = False,
    mimi_report: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    manifest_path = root / "manifests" / dataset / "episodes.jsonl"
    records = read_jsonl(manifest_path)
    if not records:
        raise ValueError("Pilot manifest is empty")
    ids = [str(record["episode_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Pilot manifest contains duplicate episode IDs")
    quality_rows = [_validate_record(root, record) for record in records]
    duration = {row["episode_id"]: row["duration_seconds"] for row in quality_rows}
    leaks = _leakage(records)
    if any(leaks.values()):
        raise ValueError(f"cross-split leakage detected: {leaks}")
    if dataset == "pilot":
        _check_canary_exclusion(root, records)
    aggregate: dict[str, dict[str, float]] = {
        "category": defaultdict(float),
        "language": defaultdict(float),
        "split": defaultdict(float),
    }
    for record in records:
        seconds = duration[record["episode_id"]]
        for dimension in aggregate:
            aggregate[dimension][str(record[dimension])] += seconds
    cross_buckets: dict[tuple[str, str, str], float] = defaultdict(float)
    for record in records:
        key = (record["category"], record["language"], record["split"])
        cross_buckets[key] += duration[record["episode_id"]]
    total = sum(duration.values())
    expected_total = dataset_spec(dataset).duration_seconds
    quota_rows = []
    for category in CATEGORIES:
        for language in LANGUAGES:
            for split in SPLITS:
                actual = cross_buckets[(category, language, split)]
                target = dataset_spec(dataset).target_seconds(category, language, split)
                error = _fraction_error(actual, target)
                quota_rows.append(
                    {
                        "dimension": "category_language_split",
                        "bucket": f"{category}/{language}/{split}",
                        "actual_seconds": actual,
                        "target_seconds": target,
                        "relative_error": error,
                        "tolerance": 0.02,
                    }
                )
                if not fixture and dataset == "pilot" and error > 0.02:
                    raise ValueError(
                        f"quota gate failed for {category}/{language}/{split}: {error:.3%}"
                    )
    for dimension, fractions in (
        ("category", CATEGORY_FRACTIONS),
        ("language", LANGUAGE_FRACTIONS),
        ("split", SPLIT_FRACTIONS),
    ):
        tolerance = 0.005 if dimension == "split" else 0.02
        for bucket, fraction in fractions.items():
            actual = aggregate[dimension][bucket]
            target = expected_total * fraction
            error = _fraction_error(actual, target)
            quota_rows.append(
                {
                    "dimension": dimension,
                    "bucket": bucket,
                    "actual_seconds": actual,
                    "target_seconds": target,
                    "relative_error": error,
                    "tolerance": tolerance,
                }
            )
            if not fixture and error > tolerance:
                raise ValueError(f"quota gate failed for {dimension}={bucket}: {error:.3%}")
    if not fixture and _fraction_error(total, expected_total) > 0.005:
        raise ValueError("total dataset duration differs from its target by more than 0.5%")
    starts = sum(int(row["starts"]) for row in quality_rows)
    stops = sum(int(row["stops"]) for row in quality_rows)
    if not fixture and dataset == "pilot" and (starts < 2_000 or stops < 2_000):
        raise ValueError("Pilot requires at least 2,000 START and STOP labels")
    synthesis_path = root / "synthesized" / f"{dataset}-utterances.jsonl"
    synthesis = read_jsonl(synthesis_path)
    scores: dict[str, list[float]] = defaultdict(list)
    for receipt in synthesis:
        if not fixture:
            require_sha256(receipt.get("model_sha256"), "TTS model")
            require_sha256(receipt.get("voice_prompt_sha256"), "voice prompt")
            normalization = receipt.get("normalization", {})
            loudness = float(normalization["integrated_lufs"])
            if not -24.0 <= loudness <= -22.0:
                raise ValueError("synthesis receipt is outside -23 +/- 1 LUFS")
            require_sha256(normalization.get("metrics_sha256"), "loudness metrics")
        require_sha256(receipt.get("recipe_sha256"), "synthesis recipe")
        if float(receipt["asr_score"]) > 0.20:
            raise ValueError("synthesis ledger retains an utterance above the 20% ASR gate")
        scores[str(receipt["asr_metric"])].append(float(receipt["asr_score"]))
    asr = {metric: sum(values) / len(values) for metric, values in scores.items()}
    if not fixture and (asr.get("cer", 1.0) > 0.08 or asr.get("wer", 1.0) > 0.08):
        raise ValueError("aggregate synthetic CER/WER exceeds 8%")
    if not fixture:
        fetch_report = read_json(root / "reports" / "fetch-report.json")
        for source in fetch_report.get("sources", []):
            require_sha256(source.get("archive_sha256"), "source archive")
            require_sha256(source.get("license_sha256"), "source license")
        require_sha256(sha256_file(root / "text" / f"{dataset}-plans.json"), "text plan")
    reviews = _review_report(root, dataset, records, duration, fixture)
    mimi = _mimi_report(root, dataset, fixture, mimi_report)
    manifest_hash = sha256_file(manifest_path)
    if not fixture and mimi.get("manifest_sha256") != manifest_hash:
        raise ValueError("Mimi decode-check report does not cover this manifest")
    license_records = {}
    for record in records:
        key = (record["source"], record["source_version"], record["license_sha256"])
        license_records[key] = {
            "source": record["source"],
            "source_version": record["source_version"],
            "source_url": record["source_url"],
            "source_license": record["source_license"],
            "license_sha256": record["license_sha256"],
            "redistribution_allowed": record["redistribution_allowed"],
        }
    reports = root / "reports" / dataset
    reports.mkdir(parents=True, exist_ok=True)
    license_report = {"licenses": list(license_records.values())}
    quota_report = {"dataset": dataset, "total_seconds": total, "buckets": quota_rows}
    quality_report = {
        "episodes": len(records),
        "starts": starts,
        "stops": stops,
        "asr": asr,
        "leakage": leaks,
        "reviews": reviews,
        "mimi_decode": mimi,
        "rows": quality_rows,
    }
    write_json(reports / "license-report.json", license_report)
    write_json(reports / "quota-report.json", quota_report)
    write_json(reports / "quality-report.json", quality_report)
    write_json(reports / "manifest-hash.json", {"sha256": manifest_hash})
    data_card = (
        f"# E2 {dataset.title()} Data Card\n\n"
        f"- Episodes: {len(records)}\n"
        f"- Timeline hours: {total / 3600:.4f}\n"
        f"- Chinese/English target: 80%/20%\n"
        f"- Sources: 30% public speech, 50% synthetic computer dialogue, "
        f"15% adjacent turns, 5% screen tasks\n"
        f"- Computer-assistant timeline: approximately 55%\n"
        f"- Runtime speech path: direct codec generation; no TTS\n"
        f"- E2 exclusions: playback echo, noise augmentation, overlap, interruption, "
        f"and feedback-loop data\n"
        f"- Manifest SHA-256: `{manifest_hash}`\n"
        f"- Audit recipe SHA-256: `{stable_hash({'dataset': dataset, 'fixture': fixture})}`\n"
    )
    (reports / "data-card.md").write_text(data_card, encoding="utf-8")
    result = {
        "dataset": dataset,
        "fixture": fixture,
        "episodes": len(records),
        "duration_seconds": total,
        "manifest_sha256": manifest_hash,
        "reports": str(reports),
        "passed": True,
    }
    write_json(root / "reports" / f"{dataset}-audit.json", result)
    return result
