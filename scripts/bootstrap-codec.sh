#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv sync --project "$ROOT/codec" --frozen
printf 'Codec environment ready. Start it with scripts/codec-worker.sh.\n'
