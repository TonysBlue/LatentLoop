#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 1
fi

uv python install 3.11
uv sync --extra ray
mkdir -p "${LATENTLOOP_DATA_ROOT:-$HOME/latentloop-data/datasets}" \
  "${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"/{checkpoints,runs,backups}
uv run latentloop inspect-model --config configs/smoke.yaml
