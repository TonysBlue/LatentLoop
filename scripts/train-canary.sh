#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
ROOT="${LATENTLOOP_DATA_ROOT:-$STORAGE_ROOT/datasets}"
CFG="${CANARY_CONFIG:-$REPO/configs/canary.yaml}"
RUN_ID="${CANARY_RUN_ID:-default}"
RUN_ROOT="${LATENTLOOP_EXPERIMENT_ROOT:-$STORAGE_ROOT/experiments}/canary/$RUN_ID"
INIT_CHECKPOINT="${CANARY_INIT_CHECKPOINT:-}"
MAX_UPDATES="${CANARY_MAX_UPDATES:-2000}"
TRACKING_MODE="${CANARY_TRACKING_MODE:-online}"
STAGE_START="${CANARY_STAGE_START:-1}"
STAGE_TOTAL="${CANARY_STAGE_TOTAL:-2}"
TRAIN_REPORT="$RUN_ROOT/runs/training.json"
SUMMARY="$REPO/tools/curation/summarize_canary.py"

source "$REPO/scripts/lib/canary-stages.sh"
canary_init_logs

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
  --set "data.shards=$ROOT/canary/v1/shards/processed/train/train-*.tar"
  --set "data.manifest=$ROOT/canary/v1/shards/processed/train/train-manifest.jsonl"
  --set "runtime.experiment_root=$RUN_ROOT"
  --set "training.max_updates=$MAX_UPDATES"
  --set "training.checkpoint_every=$MAX_UPDATES"
  --set "tracking.mode=$TRACKING_MODE"
)
if [[ -n "$INIT_CHECKPOINT" ]]; then
  readiness+=(--checkpoint "$INIT_CHECKPOINT")
  train+=(--init-from "$INIT_CHECKPOINT")
fi

mkdir -p "$RUN_ROOT"

printf -v CHECKPOINT_NAME 'step-%08d.pt' "$MAX_UPDATES"
LATEST="$RUN_ROOT/checkpoints/$CHECKPOINT_NAME"

run_training() {
  "${readiness[@]}"
  "${train[@]}" --report "$TRAIN_REPORT"
  [[ -f "$LATEST" ]] || {
    printf 'Canary training did not produce the expected checkpoint: %s\n' "$LATEST" >&2
    exit 2
  }
}

summarize_training() {
  python3 "$SUMMARY" train --report "$TRAIN_REPORT" --checkpoint "$LATEST"
}

run_evaluation() {
  local split
  for split in validation test; do
    uv run latentloop evaluate-canary \
      --config "$CFG" \
      --set "data.shards=$ROOT/canary/v1/shards/processed/$split/$split-*.tar" \
      --set "data.manifest=$ROOT/canary/v1/shards/processed/$split/$split-manifest.jsonl" \
      --checkpoint "$LATEST" --split "$split" \
      --report "$RUN_ROOT/runs/${split}-evaluation.json"
  done
}

summarize_evaluation() {
  python3 "$SUMMARY" evaluate --run-root "$RUN_ROOT" --max-updates "$MAX_UPDATES"
}

canary_stage_run "$STAGE_START" "$STAGE_TOTAL" "Train" \
  "Verify readiness and train for $MAX_UPDATES optimizer updates" \
  train run_training summarize_training
canary_stage_run "$((STAGE_START + 1))" "$STAGE_TOTAL" "Evaluate" \
  "Evaluate the checkpoint on the validation and test splits" \
  evaluate run_evaluation summarize_evaluation
