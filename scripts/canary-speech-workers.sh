#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${LATENTLOOP_MODEL_ROOT:-$HOME/latentloop-data/models}"
SOURCE_ROOT="${LATENTLOOP_VENDOR_ROOT:-$HOME/latentloop-data/vendor}"
RUN_DIR="${LATENTLOOP_RUN_DIR:-$HOME/latentloop-data/run}"
LOG_DIR="${LATENTLOOP_LOG_DIR:-$HOME/latentloop-data/logs}"
ACTION="${1:-start}"
mkdir -p "$RUN_DIR" "$LOG_DIR"

stop_worker() {
  local name="$1"
  local pid_file="$RUN_DIR/$name.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    rm -f "$pid_file"
  fi
  rm -f "$RUN_DIR/$name.sock"
}

if [[ "$ACTION" == "stop" ]]; then
  stop_worker cosyvoice
  stop_worker sensevoice
  exit 0
fi
[[ "$ACTION" == "start" ]] || { printf 'usage: %s {start|stop}\n' "$0" >&2; exit 2; }

stop_worker cosyvoice
stop_worker sensevoice
nohup setsid uv run --project "$REPO/tools/cosyvoice" python \
  "$REPO/tools/cosyvoice/worker.py" \
  --model "$MODEL_ROOT/CosyVoice2-0.5B" \
  --source "$SOURCE_ROOT/CosyVoice" \
  --socket "$RUN_DIR/cosyvoice.sock" \
  >"$LOG_DIR/cosyvoice.log" 2>&1 &
echo $! >"$RUN_DIR/cosyvoice.pid"
nohup setsid uv run --project "$REPO/tools/asr" python "$REPO/tools/asr/worker.py" \
  --model "$MODEL_ROOT/SenseVoiceSmall" --device cpu \
  --socket "$RUN_DIR/sensevoice.sock" \
  >"$LOG_DIR/sensevoice.log" 2>&1 &
echo $! >"$RUN_DIR/sensevoice.pid"

for socket in "$RUN_DIR/cosyvoice.sock" "$RUN_DIR/sensevoice.sock"; do
  for _ in $(seq 1 180); do
    [[ -S "$socket" ]] && break
    sleep 2
  done
  [[ -S "$socket" ]] || {
    printf 'Worker did not become ready: %s\n' "$socket" >&2
    tail -50 "$LOG_DIR/$(basename "$socket" .sock).log" >&2
    exit 2
  }
done
printf 'CosyVoice and SenseVoice workers are ready.\n'
