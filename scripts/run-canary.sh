#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
ROOT="${LATENTLOOP_DATA_ROOT:-$STORAGE_ROOT/datasets}"
CFG="${CANARY_CONFIG:-$REPO/configs/canary.yaml}"
LOCK="${LATENTLOOP_SOURCE_LOCK:-$ROOT/registry/source-lock.json}"
VOICES="${LATENTLOOP_VOICE_LIBRARY:-$ROOT/registry/voices/voice-library.json}"
RUNTIME_ROOT="${LATENTLOOP_RUNTIME_ROOT:-$STORAGE_ROOT/runtime}"
RUN_DIR="$RUNTIME_ROOT/sockets"
SUMMARY="$REPO/tools/curation/summarize_canary.py"
TTS_HASH="b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"
NORMALIZER="uv run --project $REPO/tools/curation python $REPO/tools/curation/normalize_adapter.py"
SCREEN="uv run --project $REPO/tools/curation python $REPO/tools/curation/screen_adapter.py"
TTS="env COSYVOICE_SOCKET=$RUN_DIR/cosyvoice.sock uv run --project $REPO/tools/cosyvoice python $REPO/tools/cosyvoice/adapter.py"
ASR="env SENSEVOICE_SOCKET=$RUN_DIR/sensevoice.sock uv run --project $REPO/tools/asr python $REPO/tools/asr/adapter.py"
ACTION="${1:-all}"

source "$REPO/scripts/lib/canary-stages.sh"
canary_init_logs

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
  uv run latentloop benchmark-codec --config "$CFG" --socket "$RUN_DIR/mimi.sock" \
    --report "$ROOT/canary/v1/reports/codec-benchmark.json"
  uv run python "$REPO/tools/curation/finalize_canary.py" \
    --config "$CFG" --root "$ROOT" --socket "$RUN_DIR/mimi.sock"
  uv run latentloop check-pilot-readiness --config "$CFG" --root "$ROOT" --dataset canary
  "$REPO/scripts/canary-mimi-worker.sh" stop
  trap - EXIT
}

bootstrap_assets() {
  "$REPO/scripts/bootstrap-canary.sh"
  "$REPO/scripts/bootstrap-canary-models.sh"
}

summarize_bootstrap() {
  printf '  Sources: public archives and licenses verified\n'
  printf '  Models: CosyVoice2 and SenseVoice assets verified\n'
}

summarize_prepare() {
  python3 "$SUMMARY" prepare --root "$ROOT"
}

summarize_encode() {
  python3 "$SUMMARY" encode --root "$ROOT"
}

case "$ACTION" in
  bootstrap)
    canary_stage_run 1 1 "Bootstrap Canary assets" \
      "Download or verify public sources, licenses, TTS, and ASR models" \
      bootstrap bootstrap_assets summarize_bootstrap
    ;;
  prepare)
    canary_stage_run 1 1 "Prepare Canary data" \
      "Verify sources, synthesize speech, run ASR gates, and build the episode manifest" \
      prepare prepare_cpu_and_speech summarize_prepare
    ;;
  encode)
    canary_stage_run 1 1 "Encode with Mimi" \
      "Benchmark Mimi, check decoded audio, audit data, encode splits, and verify readiness" \
      encode encode_and_audit summarize_encode
    ;;
  train)
    CANARY_STAGE_START=1 CANARY_STAGE_TOTAL=2 "$REPO/scripts/train-canary.sh"
    ;;
  all)
    canary_stage_run 1 4 "Prepare Canary data" \
      "Verify sources, synthesize speech, run ASR gates, and build the episode manifest" \
      prepare prepare_cpu_and_speech summarize_prepare
    canary_stage_run 2 4 "Encode with Mimi" \
      "Benchmark Mimi, check decoded audio, audit data, encode splits, and verify readiness" \
      encode encode_and_audit summarize_encode
    CANARY_STAGE_START=3 CANARY_STAGE_TOTAL=4 "$REPO/scripts/train-canary.sh"
    ;;
  *) printf 'usage: %s {bootstrap|prepare|encode|train|all}\n' "$0" >&2; exit 2 ;;
esac
