from __future__ import annotations

import argparse
from pathlib import Path

from data.curation.audit import audit_pilot_data
from data.curation.prepare import (
    check_mimi_decode,
    codec_client,
    encode_pilot_shards,
)
from runtime.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--dataset", choices=("canary", "pilot"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    client = codec_client(config, args.socket)
    client.health()
    mimi = check_mimi_decode(args.root, dataset=args.dataset, client=client)
    audit_pilot_data(
        args.root, dataset=args.dataset, mimi_report=mimi["path"]
    )
    encode_pilot_shards(args.root, dataset=args.dataset, config=config, client=client)


if __name__ == "__main__":
    main()
