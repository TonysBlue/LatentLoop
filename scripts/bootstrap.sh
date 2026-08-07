#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 1
fi

uv python install 3.11
uv sync --extra ray
STORAGE_ROOT="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
mkdir -p \
  "${LATENTLOOP_ASSET_ROOT:-$STORAGE_ROOT/assets}"/{sources,models,vendor} \
  "${LATENTLOOP_DATA_ROOT:-$STORAGE_ROOT/datasets}" \
  "${LATENTLOOP_EXPERIMENT_ROOT:-$STORAGE_ROOT/experiments}" \
  "${LATENTLOOP_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints}"/{base,smoke,released} \
  "${LATENTLOOP_RUNTIME_ROOT:-$STORAGE_ROOT/runtime}"/{sockets,logs} \
  "${LATENTLOOP_TRACKING_ROOT:-$STORAGE_ROOT/tracking}"/wandb \
  "$STORAGE_ROOT/archive" "$STORAGE_ROOT/backups"
uv run latentloop inspect-model --config configs/smoke.yaml
