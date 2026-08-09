from __future__ import annotations

import argparse

from runtime.config import load_config

from training.api import run, run_recipe
from training.evaluation import build_evaluation_report, evaluate_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--init-from")
    train_parser.add_argument("--stop-after-updates", type=int)
    recipe_parser = subparsers.add_parser("run-recipe")
    recipe_parser.add_argument("--recipe", required=True)
    recipe_parser.add_argument("--run-id")
    recipe_parser.add_argument("--set", action="append", default=[], dest="overrides")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate_parser.add_argument("--device")
    evaluate_parser.add_argument("--codec-threshold", type=float, default=0.9)
    args = parser.parse_args(argv)
    if args.command == "train":
        run(
            load_config(args.config),
            resume=args.resume,
            init_from=args.init_from,
            stop_after_updates=args.stop_after_updates,
        )
    elif args.command == "run-recipe":
        run_recipe(args.recipe, overrides=args.overrides, run_id=args.run_id)
    else:
        config = load_config(args.config)
        result = evaluate_checkpoint(
            config, args.checkpoint, split=args.split,
            device=args.device, codec_threshold=args.codec_threshold,
        )
        print(build_evaluation_report(config, args.checkpoint, args.split, result))
    return 0
