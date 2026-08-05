from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from latentloop.codec import CodecIdentity
from latentloop.codec_worker import CodecWorkerClient
from latentloop.config import ProjectConfig
from latentloop.data.codec_targets import encode_target_speech
from latentloop.data.pilot.audit import audit_pilot_data
from latentloop.data.pilot.common import (
    SPLITS,
    ensure_tree,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from latentloop.data.pilot.fetch import fetch_pilot_data
from latentloop.data.pilot.manifest import build_pilot_manifest, build_source_inventory
from latentloop.data.pilot.synthesis import synthesize_pilot
from latentloop.data.pilot.text import build_pilot_text
from latentloop.data.pilot.voices import select_pilot_voices
from latentloop.data.speech_import import import_speech_manifest
from latentloop.data.webdataset import EpisodeShardReader, write_episode_shards


def codec_client(config: ProjectConfig, socket_path: str | Path) -> CodecWorkerClient:
    return CodecWorkerClient(
        socket_path,
        CodecIdentity(
            codec_id=config.data.codec_id,
            weight_sha256=config.data.codec_weight_hash,
            revision=config.data.codec_revision,
            sample_rate=config.data.audio_sample_rate,
            frame_rate=config.data.codec_frame_rate,
            frame_samples=config.data.unit_audio_samples,
            codebooks=config.data.codec_codebooks,
            codebook_size=config.data.codec_codebook_size,
        ),
        timeout_seconds=120.0,
    )


def check_mimi_decode(
    root: str | Path,
    *,
    dataset: str,
    client: CodecWorkerClient,
    fixture: bool = False,
    minimum_segments: int = 100,
) -> dict[str, Any]:
    """Encode/decode target segments and write the report consumed by the audit gate."""
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifests" / dataset / "episodes.jsonl"
    records = read_jsonl(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    checked = 0
    failed = 0
    for record in records:
        target_path = Path(record["target_speech"])
        if not target_path.is_absolute():
            target_path = root / target_path
        waveform = torch.from_numpy(
            sf.read(target_path, dtype="float32", always_2d=False)[0]
        ).float()
        for index, segment in enumerate(record.get("target_segments", [])):
            start = int(segment["start_sample"])
            end = int(segment["end_sample"])
            session = f"mimi-check-{dataset}-{record['episode_id']}-{index}"
            try:
                encode_session = f"{session}-encode"
                decode_session = f"{session}-decode"
                client.reset(encode_session, replay=False)
                codes_for_segment = []
                for offset in range(start, end, client.identity.frame_samples):
                    frame = waveform[offset : offset + client.identity.frame_samples]
                    if frame.numel() < client.identity.frame_samples:
                        frame = torch.nn.functional.pad(
                            frame, (0, client.identity.frame_samples - frame.numel())
                        )
                    codes_for_segment.append(
                        client.encode_step(frame.reshape(1, 1, -1), encode_session)
                    )
                client.reset(decode_session, replay=False)
                for codes in codes_for_segment:
                    decoded = client.decode_step(codes, decode_session)
                    if not torch.isfinite(decoded).all():
                        raise ValueError("Mimi decoder returned non-finite samples")
                checked += 1
            except (RuntimeError, ValueError, OSError):
                failed += 1
    if not fixture and checked < minimum_segments:
        raise ValueError(
            f"Mimi decode-check requires at least {minimum_segments} successful target segments; "
            f"got {checked}"
        )
    report = {
        "dataset": dataset,
        "manifest_sha256": manifest_hash,
        "checked_segments": checked,
        "failed_segments": failed,
        "mimi_weight_sha256": client.identity.weight_sha256,
        "codec_id": client.identity.codec_id,
        "codec_revision": client.identity.revision,
        "fixture": fixture,
    }
    path = root / "reports" / f"{dataset}-mimi-decode.json"
    write_json(path, report)
    return {**report, "path": str(path)}


def encode_pilot_shards(
    root: str | Path,
    *,
    dataset: str,
    config: ProjectConfig,
    client: CodecWorkerClient,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    results: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest = root / "manifests" / dataset / f"{split}.jsonl"
        staging = root / "staging" / dataset / split / f"{split}-%06d.tar"
        processed = root / "processed" / dataset / split / f"{split}-%06d.tar"
        write_episode_shards(
            import_speech_manifest(manifest, config.data, config.model), staging
        )
        staged = EpisodeShardReader(
            str(staging).replace("%06d", "*"),
            config.data,
            config.model,
            require_encoded_speech=False,
            validate_manifest=False,
        )
        write_episode_shards(encode_target_speech(staged, client), processed)
        encoded = EpisodeShardReader(
            str(processed).replace("%06d", "*"),
            config.data,
            config.model,
            require_encoded_speech=True,
            validate_manifest=False,
        )
        episodes = units = 0
        for episode in encoded:
            episodes += 1
            units += len(episode.units)
        results.append(
            {
                "split": split,
                "episodes": episodes,
                "units": units,
                "staging": str(staging),
                "processed": str(processed),
            }
        )
    report = {"dataset": dataset, "splits": results, "codec_id": config.data.codec_id}
    write_json(root / "reports" / f"{dataset}-encoded-report.json", report)
    return report


def prepare_e2_pilot(
    root: str | Path,
    *,
    config: ProjectConfig,
    fixture: bool = False,
    lock_path: str | Path | None = None,
    download: bool = False,
    extract: bool = False,
    library: str | Path | None = None,
    synth_command: str | None = None,
    asr_command: str | None = None,
    model_sha256: str | None = None,
    normalize_command: str | None = None,
    screen_command: str | None = None,
    socket_path: str | Path | None = None,
    encode: bool = False,
    mimi_report_dir: str | Path | None = None,
    dataset: str = "all",
) -> dict[str, Any]:
    """Run all deterministic E2 preparation stages in dependency order."""
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    fetch = fetch_pilot_data(
        root, fixture=fixture, lock_path=lock_path, download=download, extract=extract
    )
    inventory = None
    if not fixture:
        if not normalize_command:
            raise ValueError("production preparation requires --normalize-command")
        inventory = build_source_inventory(normalize_command, root)
    voices = select_pilot_voices(root, library=library, fixture=fixture)
    if dataset not in {"canary", "pilot", "all"}:
        raise ValueError("dataset must be canary, pilot, or all")
    datasets: dict[str, Any] = {}
    client = codec_client(config, socket_path) if socket_path else None
    if encode and client is None:
        raise ValueError("--encode requires --socket")
    dataset_names = ("canary", "pilot") if dataset == "all" else (dataset,)
    for dataset_name in dataset_names:
        text = build_pilot_text(
            root, dataset=dataset_name, fixture=fixture, seed=config.data.seed
        )
        synthesis = synthesize_pilot(
            root,
            dataset=dataset_name,
            fixture=fixture,
            synth_command=synth_command,
            asr_command=asr_command,
            model_sha256=model_sha256,
        )
        manifest = build_pilot_manifest(
            root,
            dataset=dataset_name,
            fixture=fixture,
            normalize_command=normalize_command,
            screen_command=screen_command,
        )
        mimi_path: str | None = None
        if client:
            client.health()
            mimi = check_mimi_decode(
                root, dataset=dataset_name, client=client, fixture=fixture
            )
            mimi_path = str(mimi["path"])
        else:
            report_dir = Path(mimi_report_dir).expanduser() if mimi_report_dir else root / "reports"
            candidate = report_dir / f"{dataset_name}-mimi-decode.json"
            manifest_path = root / "manifests" / dataset_name / "episodes.jsonl"
            current_manifest_hash = sha256_file(manifest_path)
            provided = False
            if candidate.is_file():
                provided = read_json(candidate).get("manifest_sha256") == current_manifest_hash
            mimi = {"path": str(candidate), "provided": provided}
            mimi_path = str(candidate) if provided else None
        audit = (
            audit_pilot_data(
                root,
                dataset=dataset_name,
                fixture=fixture,
                mimi_report=mimi_path,
            )
            if fixture or mimi_path
            else None
        )
        encoded = (
            encode_pilot_shards(root, dataset=dataset_name, config=config, client=client)
            if encode
            else None
        )
        datasets[dataset_name] = {
            "text": text,
            "synthesis": synthesis,
            "manifest": manifest,
            "mimi": mimi,
            "audit": audit,
            "encoded": encoded,
        }
    result = {
        "root": str(root),
        "fetch": fetch,
        "voices": voices,
        "inventory": inventory,
        "datasets": datasets,
    }
    write_json(root / "reports" / "prepare-report.json", result)
    return result
