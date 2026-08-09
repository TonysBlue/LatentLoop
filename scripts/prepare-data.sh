#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
ROOT="${LATENTLOOP_DATA_ROOT:-$STORAGE_ROOT/datasets}"
DATASET="${1:-canary}"
ACTION="${2:-all}"
CFG="${DATA_CONFIG:-$REPO/configs/${DATASET}.yaml}"
RUN_DIR="${LATENTLOOP_RUNTIME_ROOT:-$STORAGE_ROOT/runtime}/sockets"
LOCK="${LATENTLOOP_SOURCE_LOCK:-$ROOT/registry/source-lock.json}"
VOICES="${LATENTLOOP_VOICE_LIBRARY:-$ROOT/registry/voices/voice-library.json}"
TTS_HASH="b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"
NORMALIZER="uv run --project $REPO/tools/curation python $REPO/tools/curation/normalize_adapter.py"
SCREEN="uv run --project $REPO/tools/curation python $REPO/tools/curation/screen_adapter.py"
TTS="env COSYVOICE_SOCKET=$RUN_DIR/cosyvoice.sock uv run --project $REPO/tools/cosyvoice python $REPO/tools/cosyvoice/adapter.py"
ASR="env SENSEVOICE_SOCKET=$RUN_DIR/sensevoice.sock uv run --project $REPO/tools/asr python $REPO/tools/asr/adapter.py"

[[ "$DATASET" == canary || "$DATASET" == pilot || "$DATASET" == production ]] || {
  printf 'data preparation supports canary, pilot, or production, got %s\n' "$DATASET" >&2
  exit 2
}
[[ -f "$CFG" ]] || { printf 'config is absent: %s\n' "$CFG" >&2; exit 2; }

prepare() {
  "$REPO/scripts/bootstrap-canary.sh"
  "$REPO/scripts/bootstrap-canary-models.sh"
  "$REPO/scripts/canary-mimi-worker.sh" stop
  "$REPO/scripts/canary-speech-workers.sh" start
  trap '"$REPO/scripts/canary-speech-workers.sh" stop' EXIT
  uv run data prepare-pilot-data \
    --config "$CFG" --root "$ROOT" --dataset "$DATASET" \
    --lock "$LOCK" --download --extract --library "$VOICES" \
    --synth-command "$TTS" --asr-command "$ASR" --model-sha256 "$TTS_HASH" \
    --normalize-command "$NORMALIZER" --screen-command "$SCREEN"
  "$REPO/scripts/canary-speech-workers.sh" stop
  trap - EXIT
}

encode() {
  "$REPO/scripts/download-mimi.sh"
  "$REPO/scripts/bootstrap-codec.sh"
  "$REPO/scripts/canary-speech-workers.sh" stop
  "$REPO/scripts/canary-mimi-worker.sh" start
  trap '"$REPO/scripts/canary-mimi-worker.sh" stop' EXIT
  uv run data benchmark-codec \
    --config "$CFG" --socket "$RUN_DIR/mimi.sock" \
    --report "$ROOT/$DATASET/v1/reports/codec-benchmark.json"
  uv run python "$REPO/tools/curation/finalize_data.py" \
    --config "$CFG" --root "$ROOT" --dataset "$DATASET" --socket "$RUN_DIR/mimi.sock"
  uv run data check-readiness --config "$CFG" --root "$ROOT"
  "$REPO/scripts/canary-mimi-worker.sh" stop
  trap - EXIT
}

rebuild_v5() {
  "$REPO/scripts/download-mimi.sh"
  "$REPO/scripts/bootstrap-codec.sh"
  "$REPO/scripts/canary-mimi-worker.sh" start
  trap '"$REPO/scripts/canary-mimi-worker.sh" stop' EXIT
  uv run data rebuild-v5 --config "$CFG" --root "$ROOT" --dataset "$DATASET" \
    --socket "$RUN_DIR/mimi.sock" --activate
  uv run data check-readiness --config "$CFG" --root "$ROOT"
  "$REPO/scripts/canary-mimi-worker.sh" stop
  trap - EXIT
}

case "$ACTION" in
  bootstrap) "$REPO/scripts/bootstrap-canary.sh"; "$REPO/scripts/bootstrap-canary-models.sh" ;;
  prepare) prepare ;;
  encode) encode ;;
  rebuild-v5) rebuild_v5 ;;
  all) prepare; encode ;;
  *) printf 'usage: %s {canary|pilot|production} {bootstrap|prepare|encode|rebuild-v5|all}\n' "$0" >&2; exit 2 ;;
esac
