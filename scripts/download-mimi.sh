#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${MIMI_MODEL_DIR:-$HOME/latentloop-data/models/mimi}"
EXPECTED="09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
FILENAME="tokenizer-e351c8d8-checkpoint125.safetensors"

mkdir -p "$TARGET"
uv run --project "$ROOT/codec" --frozen python - "$TARGET" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(
    "kyutai/moshika-pytorch-bf16",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    revision="a49141e28b3d9c947cf9aa5314431e1b11cbd2f5",
    local_dir=sys.argv[1],
)
PY

ACTUAL="$(sha256sum "$TARGET/$FILENAME" | cut -d' ' -f1)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  printf 'Mimi SHA-256 mismatch: expected %s, got %s\n' "$EXPECTED" "$ACTUAL" >&2
  exit 1
fi
printf '%s\n' "$TARGET/$FILENAME"
