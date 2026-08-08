from __future__ import annotations

import argparse
import json

from data.migrate import migrate_manifest_v4_to_v5
from latentloop.config import load_config
from latentloop.data.curation import check_readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate-manifest")
    migrate.add_argument("--source", required=True)
    migrate.add_argument("--destination", required=True)
    readiness = subparsers.add_parser("check-readiness")
    readiness.add_argument("--config", required=True)
    readiness.add_argument("--root")
    args = parser.parse_args(argv)
    if args.command == "migrate-manifest":
        source_hash = migrate_manifest_v4_to_v5(args.source, args.destination)
        print(json.dumps({"source_sha256": source_hash, "destination": args.destination}))
    else:
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        print(json.dumps(check_readiness(root, config=config), indent=2))
    return 0
