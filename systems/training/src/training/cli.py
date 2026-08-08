from __future__ import annotations

import argparse

from latentloop.config import load_config
from training.api import run, run_recipe


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
    args = parser.parse_args(argv)
    if args.command == "train":
        run(
            load_config(args.config),
            resume=args.resume,
            init_from=args.init_from,
            stop_after_updates=args.stop_after_updates,
        )
    else:
        run_recipe(args.recipe, overrides=args.overrides, run_id=args.run_id)
    return 0
