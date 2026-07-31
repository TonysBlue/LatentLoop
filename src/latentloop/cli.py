from __future__ import annotations

import argparse
import json
import sys

import torch

from latentloop.config import ProjectConfig, load_config
from latentloop.data import EpisodeShardReader, SyntheticEpisodeDataset, write_episode_shards
from latentloop.model import StreamingLatentLoop
from latentloop.ray_jobs import generate_synthetic_with_ray, write_ray_report
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
        loss = output.speech_logits.float().mean() + output.memory_logits.float().mean()
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
    output = args.output or str(config.runtime.root_path() / "processed" / "train-%06d.tar")
    if args.ray:
        manifest = generate_synthetic_with_ray(config, output)
        write_ray_report(config.runtime.root_path() / "runs" / "ray-generate.json", manifest)
    else:
        manifest = write_episode_shards(SyntheticEpisodeDataset(config.data, config.model), output)
    print(json.dumps({"episodes": len(manifest), "output": output}, indent=2))
    return 0


def validate_data(args: argparse.Namespace) -> int:
    config = _load(args)
    source = config.data.shards if config.data.source == "webdataset" else args.shards
    if not source:
        raise ValueError("provide --shards or configure a webdataset source")
    episodes = 0
    units = 0
    for episode in EpisodeShardReader(source, config.data, config.model):
        episodes += 1
        units += len(episode.units)
    print(json.dumps({"episodes": episodes, "units": units}, indent=2))
    return 0


def train_command(args: argparse.Namespace) -> int:
    config = _load(args)
    result = train(config, resume=args.resume)
    print(
        json.dumps(
            {"train_state": result["train_state"], "metrics": result["metrics"]},
            indent=2,
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

    validate_parser = subparsers.add_parser("validate-data")
    _config_arguments(validate_parser)
    validate_parser.add_argument("--shards", help="Glob for generated shards")
    validate_parser.set_defaults(handler=validate_data)

    train_parser = subparsers.add_parser("train")
    _config_arguments(train_parser)
    train_parser.add_argument("--resume", help="Checkpoint path")
    train_parser.set_defaults(handler=train_command)
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
