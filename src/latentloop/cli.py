from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from latentloop.codec import CodecIdentity
from latentloop.codec_worker import CodecWorkerClient
from latentloop.config import ProjectConfig, load_config
from latentloop.data import (
    EpisodeShardReader,
    SpeechOverfitDataset,
    SyntheticEpisodeDataset,
    encode_target_speech,
    import_speech_manifest,
    write_episode_shards,
)
from latentloop.data.curation import (
    audit_pilot_data,
    build_pilot_manifest,
    build_pilot_text,
    check_readiness,
    fetch_pilot_data,
    prepare_pilot_data,
    select_pilot_voices,
    synthesize_pilot,
)
from latentloop.evaluation import build_evaluation_report, evaluate_checkpoint
from latentloop.model import StreamingLatentLoop
from latentloop.ray_jobs import generate_synthetic_with_ray, write_ray_report
from latentloop.recipe import run_recipe
from latentloop.speech_metrics import benchmark_decoder
from latentloop.training import train


def _config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML project configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, repeatable",
    )


def _load(args: argparse.Namespace) -> ProjectConfig:
    return load_config(args.config, args.overrides)


def _codec_client(config: ProjectConfig, socket_path: str) -> CodecWorkerClient:
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


def _emit_json_report(value: object, path: str | None = None) -> None:
    report = json.dumps(value, indent=2)
    if path:
        report_path = Path(path).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n", encoding="utf-8")
    print(report)


def inspect_model(args: argparse.Namespace) -> int:
    config = _load(args)
    model = StreamingLatentLoop(config.model)
    parameters = model.parameter_count()
    print(
        json.dumps(
            {
                "parameters": parameters,
                "billions": round(parameters / 1_000_000_000, 4),
                "tokens_per_unit": config.model.tokens_per_unit,
                "max_kv_tokens": config.model.tokens_per_unit * config.model.kv_units,
            },
            indent=2,
        )
    )
    return 0


def doctor(args: argparse.Namespace) -> int:
    config = _load(args)
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        report["status"] = "failed"
        print(json.dumps(report, indent=2))
        return 1
    device = torch.device("cuda:0")
    report.update(
        {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime": torch.version.cuda,
        }
    )
    try:
        model = StreamingLatentLoop(config.model).to(device=device, dtype=torch.float16)
        episode = SyntheticEpisodeDataset(config.data, config.model).make_episode(0)
        unit = episode.units[0].to(device)
        unit.mic_audio = unit.mic_audio.half()
        unit.screen = unit.screen.half()
        output = model(unit, model.initial_state(1, device))
        loss = (
            output.speech_mode_logits.float().mean()
            + output.speech_codec_logits.float().mean()
            + output.action_logits.float().mean()
        )
        loss.backward()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        report.update(
            {
                "status": "ok",
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
        exit_code = 0
    except RuntimeError as error:
        report.update({"status": "failed", "error": str(error)})
        exit_code = 1
    print(json.dumps(report, indent=2))
    return exit_code


def generate_data(args: argparse.Namespace) -> int:
    config = _load(args)
    output = args.output or str(config.runtime.data_path() / "generated" / "train-%06d.tar")
    if args.ray:
        manifest = generate_synthetic_with_ray(config, output)
        write_ray_report(config.runtime.data_path() / "generated" / "ray-report.json", manifest)
    else:
        manifest = write_episode_shards(SyntheticEpisodeDataset(config.data, config.model), output)
    print(json.dumps({"episodes": len(manifest), "output": output}, indent=2))
    return 0


def build_overfit_data(args: argparse.Namespace) -> int:
    config = _load(args)
    if config.data.train_episodes != 32:
        raise ValueError("the direct-speech overfit gate requires exactly 32 trajectories")
    manifest = write_episode_shards(SpeechOverfitDataset(config.data, config.model), args.output)
    print(json.dumps({"episodes": len(manifest), "output": args.output}, indent=2))
    return 0


def validate_data(args: argparse.Namespace) -> int:
    config = _load(args)
    source = args.shards or config.data.shards
    if not source:
        raise ValueError("provide --shards or configure a webdataset source")
    episodes = 0
    units = 0
    for episode in EpisodeShardReader(source, config.data, config.model):
        episodes += 1
        units += len(episode.units)
    print(json.dumps({"episodes": episodes, "units": units}, indent=2))
    return 0


def encode_speech(args: argparse.Namespace) -> int:
    config = _load(args)
    source = args.shards or config.data.shards
    if not source:
        raise ValueError("provide --shards or configure a WebDataset source")
    episodes = EpisodeShardReader(
        source,
        config.data,
        config.model,
        require_encoded_speech=False,
        validate_manifest=False,
    )
    client = _codec_client(config, args.socket)
    client.health()
    manifest = write_episode_shards(encode_target_speech(episodes, client), args.output)
    print(json.dumps({"episodes": len(manifest), "output": args.output}, indent=2))
    return 0


def import_speech(args: argparse.Namespace) -> int:
    config = _load(args)
    episodes = import_speech_manifest(args.manifest, config.data, config.model)
    manifest = write_episode_shards(episodes, args.output)
    print(json.dumps({"episodes": len(manifest), "output": args.output}, indent=2))
    return 0


def _pilot_root(args: argparse.Namespace, config: ProjectConfig) -> Path:
    if args.root:
        return Path(args.root).expanduser().resolve()
    if value := os.environ.get("LATENTLOOP_DATA_ROOT"):
        return Path(value).expanduser().resolve()
    return config.runtime.data_path()


def fetch_pilot_data_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = fetch_pilot_data(
        _pilot_root(args, config),
        fixture=args.fixture,
        lock_path=args.lock,
        download=args.download,
        extract=args.extract,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_pilot_text_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = build_pilot_text(
        _pilot_root(args, config),
        dataset=args.dataset,
        fixture=args.fixture,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


def select_pilot_voices_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = select_pilot_voices(
        _pilot_root(args, config),
        library=args.library,
        fixture=args.fixture,
    )
    print(json.dumps(result, indent=2))
    return 0


def synthesize_pilot_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = synthesize_pilot(
        _pilot_root(args, config),
        dataset=args.dataset,
        fixture=args.fixture,
        synth_command=args.synth_command,
        asr_command=args.asr_command,
        model_sha256=args.model_sha256,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_pilot_manifest_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = build_pilot_manifest(
        _pilot_root(args, config),
        dataset=args.dataset,
        fixture=args.fixture,
        normalize_command=args.normalize_command,
        screen_command=args.screen_command,
    )
    print(json.dumps(result, indent=2))
    return 0


def audit_pilot_data_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = audit_pilot_data(
        _pilot_root(args, config),
        dataset=args.dataset,
        fixture=args.fixture,
        mimi_report=args.mimi_report,
    )
    print(json.dumps(result, indent=2))
    return 0


def prepare_pilot_data_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = prepare_pilot_data(
        _pilot_root(args, config),
        config=config,
        fixture=args.fixture,
        lock_path=args.lock,
        download=args.download,
        extract=args.extract,
        library=args.library,
        synth_command=args.synth_command,
        asr_command=args.asr_command,
        model_sha256=args.model_sha256,
        normalize_command=args.normalize_command,
        screen_command=args.screen_command,
        socket_path=args.socket,
        encode=args.encode,
        mimi_report_dir=args.mimi_report_dir,
        dataset=args.dataset,
    )
    print(json.dumps(result, indent=2))
    return 0


def check_readiness_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = check_readiness(
        _pilot_root(args, config),
        config=config,
        dataset=config.data.dataset,
        require_checkpoint=args.checkpoint,
        require_encoded=not args.allow_unencoded,
    )
    print(json.dumps(result, indent=2))
    return 0


def benchmark_codec(args: argparse.Namespace) -> int:
    config = _load(args)
    client = _codec_client(config, args.socket)
    health = client.health()
    generator = torch.Generator().manual_seed(config.data.seed)
    codes = torch.randint(
        config.data.codec_codebook_size,
        (args.frames, config.data.codec_codebooks, 1),
        generator=generator,
    )
    result = benchmark_decoder(client, codes)
    _emit_json_report(
        {"health": health, "benchmark": asdict(result)},
        args.report,
    )
    return int(result.rtf >= 1 or result.p95_frame_ms >= config.data.unit_ms)


def benchmark_stream(args: argparse.Namespace) -> int:
    config = _load(args)
    if not torch.cuda.is_available():
        raise RuntimeError("stream benchmark requires CUDA")
    device = torch.device("cuda:0")
    client = _codec_client(config, args.socket)
    client.health()
    model = StreamingLatentLoop(config.model).to(device=device, dtype=torch.float16).eval()
    episode = SyntheticEpisodeDataset(config.data, config.model).make_episode(0)
    state = model.initial_state(1, device)
    client.reset("stream-benchmark", replay=False)
    latencies: list[float] = []
    with torch.inference_mode():
        for index in range(args.warmup + args.frames):
            unit = episode.units[index % len(episode.units)].to(device)
            unit.mic_audio = unit.mic_audio.half()
            unit.screen = unit.screen.half()
            started = time.perf_counter()
            generated = model.generate_step(unit, state)
            state = generated.output.state
            if bool(generated.speech_mode.item()):
                client.decode_step(generated.speech_codes.transpose(1, 2), "stream-benchmark")
            if index >= args.warmup:
                latencies.append(time.perf_counter() - started)
    measured = torch.tensor(latencies)
    elapsed = float(measured.sum().item())
    health = client.health()
    peak_model = torch.cuda.max_memory_allocated(device)
    codec_allocated = int(health.get("memory_allocated_bytes", 0))
    result = {
        "frames": len(latencies),
        "rtf": elapsed / (len(latencies) * config.data.unit_ms / 1_000),
        "mean_frame_ms": float(measured.mean().item() * 1_000),
        "p95_frame_ms": float(measured.quantile(0.95).item() * 1_000),
        "model_peak_memory_bytes": peak_model,
        "codec_memory_bytes": codec_allocated,
        "combined_memory_bytes": peak_model + codec_allocated,
        "kv_tokens": int(state.layer_kv[0].key.shape[2]),
    }
    print(json.dumps(result, indent=2))
    return int(
        result["p95_frame_ms"] >= config.data.unit_ms
        or result["combined_memory_bytes"] >= int(7.5 * 1024**3)
    )


def evaluate(args: argparse.Namespace) -> int:
    config = _load(args)
    result = evaluate_checkpoint(
        config,
        args.checkpoint,
        split=args.split,
        device=args.device,
        codec_threshold=args.codec_threshold,
    )
    report = build_evaluation_report(config, args.checkpoint, args.split, result)
    _emit_json_report(report, args.report)
    return int(report.get("passed") is False)


def train_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = train(config, resume=args.resume, init_from=args.init_from)
    _emit_json_report(
        {
            "train_state": result["train_state"],
            "metrics": result["metrics"],
            "tracking": result["tracking"],
        },
        args.report,
    )
    return 0


def run_recipe_command(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            run_recipe(args.recipe, args.overrides, run_id=args.run_id), indent=2, default=str
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latentloop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-model")
    _config_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=inspect_model)

    doctor_parser = subparsers.add_parser("doctor")
    _config_arguments(doctor_parser)
    doctor_parser.set_defaults(handler=doctor)

    generate_parser = subparsers.add_parser("generate-data")
    _config_arguments(generate_parser)
    generate_parser.add_argument("--output", help="ShardWriter pattern, for example train-%06d.tar")
    generate_parser.add_argument("--ray", action="store_true", help="Use CPU-only Ray workers")
    generate_parser.set_defaults(handler=generate_data)

    overfit_data_parser = subparsers.add_parser("build-overfit-data")
    _config_arguments(overfit_data_parser)
    overfit_data_parser.add_argument("--output", required=True, help="Staging ShardWriter pattern")
    overfit_data_parser.set_defaults(handler=build_overfit_data)

    validate_parser = subparsers.add_parser("validate-data")
    _config_arguments(validate_parser)
    validate_parser.add_argument("--shards", help="Glob for generated shards")
    validate_parser.set_defaults(handler=validate_data)

    encode_parser = subparsers.add_parser("encode-speech")
    _config_arguments(encode_parser)
    encode_parser.add_argument("--shards", help="Input shard glob")
    encode_parser.add_argument("--output", required=True, help="Output ShardWriter pattern")
    encode_parser.add_argument("--socket", required=True, help="Mimi worker Unix socket")
    encode_parser.set_defaults(handler=encode_speech)

    import_parser = subparsers.add_parser("import-speech")
    _config_arguments(import_parser)
    import_parser.add_argument("--manifest", required=True, help="Speech source JSONL")
    import_parser.add_argument("--output", required=True, help="Staging ShardWriter pattern")
    import_parser.set_defaults(handler=import_speech)

    fetch_pilot_parser = subparsers.add_parser("fetch-pilot-data")
    _config_arguments(fetch_pilot_parser)
    fetch_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    fetch_pilot_parser.add_argument("--lock", help="Locked production source JSON")
    fetch_pilot_parser.add_argument("--download", action="store_true")
    fetch_pilot_parser.add_argument("--extract", action="store_true")
    fetch_pilot_parser.add_argument("--fixture", action="store_true")
    fetch_pilot_parser.set_defaults(handler=fetch_pilot_data_command)

    text_pilot_parser = subparsers.add_parser("build-pilot-text")
    _config_arguments(text_pilot_parser)
    text_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    text_pilot_parser.add_argument("--dataset", choices=("canary", "pilot"), required=True)
    text_pilot_parser.add_argument("--seed", type=int, default=17)
    text_pilot_parser.add_argument("--fixture", action="store_true")
    text_pilot_parser.set_defaults(handler=build_pilot_text_command)

    voices_pilot_parser = subparsers.add_parser("select-pilot-voices")
    _config_arguments(voices_pilot_parser)
    voices_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    voices_pilot_parser.add_argument("--library", help="Authorized CosyVoice voice JSON")
    voices_pilot_parser.add_argument("--fixture", action="store_true")
    voices_pilot_parser.set_defaults(handler=select_pilot_voices_command)

    synthesize_pilot_parser = subparsers.add_parser("synthesize-pilot")
    _config_arguments(synthesize_pilot_parser)
    synthesize_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    synthesize_pilot_parser.add_argument("--dataset", choices=("canary", "pilot"), required=True)
    synthesize_pilot_parser.add_argument("--synth-command")
    synthesize_pilot_parser.add_argument("--asr-command")
    synthesize_pilot_parser.add_argument("--model-sha256")
    synthesize_pilot_parser.add_argument("--fixture", action="store_true")
    synthesize_pilot_parser.set_defaults(handler=synthesize_pilot_command)

    manifest_pilot_parser = subparsers.add_parser("build-pilot-manifest")
    _config_arguments(manifest_pilot_parser)
    manifest_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    manifest_pilot_parser.add_argument("--dataset", choices=("canary", "pilot"), required=True)
    manifest_pilot_parser.add_argument("--normalize-command")
    manifest_pilot_parser.add_argument("--screen-command")
    manifest_pilot_parser.add_argument("--fixture", action="store_true")
    manifest_pilot_parser.set_defaults(handler=build_pilot_manifest_command)

    audit_pilot_parser = subparsers.add_parser("audit-pilot-data")
    _config_arguments(audit_pilot_parser)
    audit_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    audit_pilot_parser.add_argument("--dataset", choices=("canary", "pilot"), required=True)
    audit_pilot_parser.add_argument("--mimi-report")
    audit_pilot_parser.add_argument("--fixture", action="store_true")
    audit_pilot_parser.set_defaults(handler=audit_pilot_data_command)

    prepare_pilot_parser = subparsers.add_parser(
        "prepare-pilot-data", help="Run automatic Canary/Pilot preparation and gates"
    )
    _config_arguments(prepare_pilot_parser)
    prepare_pilot_parser.add_argument("--root", help="Shared dataset artifact root")
    prepare_pilot_parser.add_argument("--lock", help="Locked production source JSON")
    prepare_pilot_parser.add_argument("--download", action="store_true")
    prepare_pilot_parser.add_argument("--extract", action="store_true")
    prepare_pilot_parser.add_argument("--library", help="Authorized CosyVoice voice JSON")
    prepare_pilot_parser.add_argument("--synth-command")
    prepare_pilot_parser.add_argument("--asr-command")
    prepare_pilot_parser.add_argument("--model-sha256")
    prepare_pilot_parser.add_argument("--normalize-command")
    prepare_pilot_parser.add_argument("--screen-command")
    prepare_pilot_parser.add_argument("--socket", help="Mimi worker Unix socket")
    prepare_pilot_parser.add_argument("--encode", action="store_true")
    prepare_pilot_parser.add_argument("--mimi-report-dir")
    prepare_pilot_parser.add_argument(
        "--dataset", choices=("canary", "pilot", "all"), default="canary"
    )
    prepare_pilot_parser.add_argument("--fixture", action="store_true")
    prepare_pilot_parser.set_defaults(handler=prepare_pilot_data_command)

    readiness_parser = subparsers.add_parser(
        "check-readiness", help="Verify automatic gates before real-data training"
    )
    _config_arguments(readiness_parser)
    readiness_parser.add_argument("--root", help="Shared dataset artifact root")
    readiness_parser.add_argument("--checkpoint", help="Optional initial checkpoint")
    readiness_parser.add_argument("--allow-unencoded", action="store_true")
    readiness_parser.set_defaults(handler=check_readiness_command)

    codec_parser = subparsers.add_parser("benchmark-codec")
    _config_arguments(codec_parser)
    codec_parser.add_argument("--socket", required=True, help="Mimi worker Unix socket")
    codec_parser.add_argument("--frames", type=int, default=250)
    codec_parser.add_argument("--report", help="Optional JSON report path")
    codec_parser.set_defaults(handler=benchmark_codec)

    stream_parser = subparsers.add_parser("benchmark-stream")
    _config_arguments(stream_parser)
    stream_parser.add_argument("--socket", required=True, help="Mimi worker Unix socket")
    stream_parser.add_argument("--frames", type=int, default=250)
    stream_parser.add_argument("--warmup", type=int, default=10)
    stream_parser.set_defaults(handler=benchmark_stream)

    evaluation_parser = subparsers.add_parser("evaluate")
    _config_arguments(evaluation_parser)
    evaluation_parser.add_argument("--checkpoint", required=True)
    evaluation_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation_parser.add_argument("--device")
    evaluation_parser.add_argument("--codec-threshold", type=float, default=0.9)
    evaluation_parser.add_argument("--report", help="Optional JSON report path")
    evaluation_parser.set_defaults(handler=evaluate)

    train_parser = subparsers.add_parser("train")
    _config_arguments(train_parser)
    train_parser.add_argument("--resume", help="Checkpoint path")
    train_parser.add_argument("--init-from", help="Warm-start compatible base weights")
    train_parser.add_argument("--report", help="Optional JSON report path")
    train_parser.set_defaults(handler=train_command)
    recipe_parser = subparsers.add_parser(
        "run-recipe", help="Run a multi-stage training recipe with shared gates and evaluation"
    )
    recipe_parser.add_argument("--recipe", required=True)
    recipe_parser.add_argument("--run-id", help="Unique artifact namespace (default: generated)")
    recipe_parser.add_argument("--set", action="append", default=[], dest="overrides")
    recipe_parser.set_defaults(handler=run_recipe_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
