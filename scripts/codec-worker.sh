#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="${MIMI_WEIGHTS:-$HOME/latentloop-data/models/mimi/tokenizer-e351c8d8-checkpoint125.safetensors}"
SOCKET="${LATENTLOOP_CODEC_SOCKET:-$HOME/latentloop-data/run/mimi.sock}"

exec uv run --project "$ROOT/codec" --frozen python "$ROOT/codec/worker.py" \
  --weights "$WEIGHTS" --socket "$SOCKET" --device "${MIMI_DEVICE:-cuda:0}"
