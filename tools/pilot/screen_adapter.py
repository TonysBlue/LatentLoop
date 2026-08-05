from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("operation") != "capture-isolated-sandbox":
        raise ValueError("unsupported screen adapter operation")
    ticks = int(request["ticks"])
    if ticks < 1:
        raise ValueError("screen timeline must contain at least one tick")
    seed = int(str(request["episode_id"]).encode().hex()[-8:], 16)
    frame = np.full((1, 3, 224, 224), 0.08, dtype=np.float32)
    left = 20 + seed % 96
    frame[:, 0, 24:52, left : left + 100] = 0.72
    frame[:, 1, 64:190, 24:200] = 0.22
    frame[:, 2, 76:178, 36:188] = 0.44
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ticks=np.asarray([min(1, ticks - 1)], dtype=np.int64),
        frames=frame,
    )


if __name__ == "__main__":
    main()
