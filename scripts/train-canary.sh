#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PILOT_DATA_ROOT:-$HOME/latentloop-data/pilot-data}"
CFG="${CANARY_CONFIG:-$REPO/configs/canary.yaml}"
RUN_ROOT="${CANARY_RUN_ROOT:-$HOME/latentloop-data/canary-run}"
INIT_CHECKPOINT="${CANARY_INIT_CHECKPOINT:-}"
MAX_UPDATES="${CANARY_MAX_UPDATES:-2000}"
TRACKING_MODE="${CANARY_TRACKING_MODE:-online}"

[[ -f "$CFG" ]] || { printf 'Canary config is absent: %s\n' "$CFG" >&2; exit 2; }
[[ -d "$ROOT" ]] || { printf 'Canary data root is absent: %s\nRun scripts/prepare-canary.sh first.\n' "$ROOT" >&2; exit 2; }

cd "$REPO"
readiness=(
  uv run latentloop check-pilot-readiness
  --config "$CFG"
  --root "$ROOT"
  --dataset canary
)
train=(
  uv run latentloop train
  --config "$CFG"
  --set "data.shards=$ROOT/processed/canary/train/train-*.tar"
  --set "data.manifest=$ROOT/processed/canary/train/train-manifest.jsonl"
  --set "runtime.data_root=$RUN_ROOT"
  --set "training.max_updates=$MAX_UPDATES"
  --set "training.checkpoint_every=$MAX_UPDATES"
  --set "tracking.mode=$TRACKING_MODE"
)
if [[ -n "$INIT_CHECKPOINT" ]]; then
  readiness+=(--checkpoint "$INIT_CHECKPOINT")
  train+=(--init-from "$INIT_CHECKPOINT")
fi

mkdir -p "$RUN_ROOT"

"${readiness[@]}"
"${train[@]}"
printf -v CHECKPOINT_NAME 'step-%08d.pt' "$MAX_UPDATES"
LATEST="$RUN_ROOT/checkpoints/$CHECKPOINT_NAME"
[[ -f "$LATEST" ]] || {
  printf 'Canary training did not produce the expected checkpoint: %s\n' "$LATEST" >&2
  exit 2
}
printf 'Canary training complete. Latest checkpoint: %s\n' "$LATEST"

for split in validation test; do
  uv run latentloop evaluate-canary \
    --config "$CFG" \
    --set "data.shards=$ROOT/processed/canary/$split/$split-*.tar" \
    --set "data.manifest=$ROOT/processed/canary/$split/$split-manifest.jsonl" \
    --checkpoint "$LATEST" --split "$split" \
    --report "$RUN_ROOT/runs/${split}-evaluation.json"
done
