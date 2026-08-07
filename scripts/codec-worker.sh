#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
ASSET_ROOT="${LATENTLOOP_ASSET_ROOT:-$STORAGE_ROOT/assets}"
RUNTIME_ROOT="${LATENTLOOP_RUNTIME_ROOT:-$STORAGE_ROOT/runtime}"
WEIGHTS="$ASSET_ROOT/models/mimi/tokenizer-e351c8d8-checkpoint125.safetensors"
SOCKET="$RUNTIME_ROOT/sockets/mimi.sock"

exec uv run --project "$ROOT/codec" --frozen python "$ROOT/codec/worker.py" \
  --weights "$WEIGHTS" --socket "$SOCKET" --device "${MIMI_DEVICE:-cuda:0}"
