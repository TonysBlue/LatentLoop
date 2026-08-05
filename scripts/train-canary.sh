#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${E2_ROOT:-$HOME/latentloop-data/e2-pilot}"
CFG="${E2_CONFIG:-$REPO/configs/e2-canary.yaml}"
RUN_ROOT="${E2_CANARY_RUN_ROOT:-$HOME/latentloop-data/e2-canary-run}"
INIT_CHECKPOINT="${E2_INIT_CHECKPOINT:-}"
MAX_UPDATES="${E2_CANARY_MAX_UPDATES:-2000}"
TRACKING_MODE="${E2_TRACKING_MODE:-online}"

[[ -f "$CFG" ]] || { printf 'Canary config is absent: %s\n' "$CFG" >&2; exit 2; }
[[ -d "$ROOT" ]] || { printf 'Canary data root is absent: %s\nRun scripts/prepare-canary.sh first.\n' "$ROOT" >&2; exit 2; }

cd "$REPO"
readiness=(
  uv run latentloop check-e2-readiness
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
