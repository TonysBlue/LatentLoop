#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${LATENTLOOP_RUN_DIR:-$HOME/latentloop-data/run}"
LOG_DIR="${LATENTLOOP_LOG_DIR:-$HOME/latentloop-data/logs}"
ACTION="${1:-start}"
mkdir -p "$RUN_DIR" "$LOG_DIR"

stop_worker() {
  local pids=()
  if [[ -f "$RUN_DIR/mimi.pid" ]]; then
    pid="$(cat "$RUN_DIR/mimi.pid")"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    rm -f "$RUN_DIR/mimi.pid"
  fi
  # Also clean workers started before process-group PID tracking was introduced.
  mapfile -t pids < <(
    pgrep -f "$REPO/codec/worker.py.*--socket $RUN_DIR/mimi.sock" || true
  )
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      mapfile -t pids < <(
        pgrep -f "$REPO/codec/worker.py.*--socket $RUN_DIR/mimi.sock" || true
      )
      ((${#pids[@]} == 0)) && break
      sleep 1
    done
    ((${#pids[@]} == 0)) || kill -9 "${pids[@]}" 2>/dev/null || true
  fi
  rm -f "$RUN_DIR/mimi.sock"
}

if [[ "$ACTION" == "stop" ]]; then
  stop_worker
  exit 0
fi
[[ "$ACTION" == "start" ]] || { printf 'usage: %s {start|stop}\n' "$0" >&2; exit 2; }
stop_worker
nohup setsid "$REPO/scripts/codec-worker.sh" >"$LOG_DIR/mimi.log" 2>&1 &
echo $! >"$RUN_DIR/mimi.pid"
for _ in $(seq 1 120); do
  [[ -S "$RUN_DIR/mimi.sock" ]] && break
  sleep 2
done
[[ -S "$RUN_DIR/mimi.sock" ]] || {
  printf 'Mimi worker did not become ready.\n' >&2
  tail -50 "$LOG_DIR/mimi.log" >&2
  exit 2
}
printf 'Mimi worker is ready.\n'
