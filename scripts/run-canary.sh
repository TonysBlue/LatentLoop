#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PILOT_DATA_ROOT:-$HOME/latentloop-data/pilot-data}"
CFG="${CANARY_CONFIG:-$REPO/configs/canary.yaml}"
LOCK="${PILOT_SOURCE_LOCK:-$ROOT/raw/source-lock.json}"
VOICES="${PILOT_VOICE_LIBRARY:-$ROOT/voices/voice-library.json}"
RUN_DIR="${LATENTLOOP_RUN_DIR:-$HOME/latentloop-data/run}"
TTS_HASH="b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"
NORMALIZER="uv run --project $REPO/tools/pilot python $REPO/tools/pilot/normalize_adapter.py"
SCREEN="uv run --project $REPO/tools/pilot python $REPO/tools/pilot/screen_adapter.py"
TTS="env COSYVOICE_SOCKET=$RUN_DIR/cosyvoice.sock uv run --project $REPO/tools/cosyvoice python $REPO/tools/cosyvoice/adapter.py"
ASR="env SENSEVOICE_SOCKET=$RUN_DIR/sensevoice.sock uv run --project $REPO/tools/asr python $REPO/tools/asr/adapter.py"
ACTION="${1:-all}"

prepare_cpu_and_speech() {
  "$REPO/scripts/bootstrap-canary.sh"
  "$REPO/scripts/bootstrap-canary-models.sh"
  "$REPO/scripts/canary-mimi-worker.sh" stop
  "$REPO/scripts/canary-speech-workers.sh" start
  trap '"$REPO/scripts/canary-speech-workers.sh" stop' EXIT
  uv run latentloop prepare-pilot-data \
    --config "$CFG" --root "$ROOT" --dataset canary \
    --lock "$LOCK" --download --extract --library "$VOICES" \
    --synth-command "$TTS" --asr-command "$ASR" --model-sha256 "$TTS_HASH" \
    --normalize-command "$NORMALIZER" --screen-command "$SCREEN"
  "$REPO/scripts/canary-speech-workers.sh" stop
  trap - EXIT
}

encode_and_audit() {
  "$REPO/scripts/download-mimi.sh"
  "$REPO/scripts/bootstrap-codec.sh"
  "$REPO/scripts/canary-speech-workers.sh" stop
  "$REPO/scripts/canary-mimi-worker.sh" start
  trap '"$REPO/scripts/canary-mimi-worker.sh" stop' EXIT
  uv run latentloop benchmark-codec --config "$CFG" --socket "$RUN_DIR/mimi.sock"
  uv run python "$REPO/tools/pilot/finalize_canary.py" \
    --config "$CFG" --root "$ROOT" --socket "$RUN_DIR/mimi.sock"
  uv run latentloop check-pilot-readiness --config "$CFG" --root "$ROOT" --dataset canary
  "$REPO/scripts/canary-mimi-worker.sh" stop
  trap - EXIT
}

case "$ACTION" in
  bootstrap) "$REPO/scripts/bootstrap-canary.sh"; "$REPO/scripts/bootstrap-canary-models.sh" ;;
  prepare) prepare_cpu_and_speech ;;
  encode) encode_and_audit ;;
  train) "$REPO/scripts/train-canary.sh" ;;
  all) prepare_cpu_and_speech; encode_and_audit; "$REPO/scripts/train-canary.sh" ;;
  *) printf 'usage: %s {bootstrap|prepare|encode|train|all}\n' "$0" >&2; exit 2 ;;
esac
