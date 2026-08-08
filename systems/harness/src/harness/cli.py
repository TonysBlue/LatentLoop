from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = Path(args.config).expanduser()
    if not config.is_file():
        raise FileNotFoundError(config)
    # A formal deployment must inject SensorAdapter, ActuatorAdapter and
    # EvaluatorAdapter.  The CLI never silently substitutes a fake backend.
    raise RuntimeError(
        "QEMU/KVM Harness requires deployment adapters; configure them through the deployment API"
    )
