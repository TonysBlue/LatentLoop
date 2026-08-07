#!/usr/bin/env bash

# Shared terminal output and logging for Canary orchestration scripts.

canary_init_logs() {
  local default_root
  local storage_root experiment_root run_id
  storage_root="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
  experiment_root="${LATENTLOOP_EXPERIMENT_ROOT:-$storage_root/experiments}"
  run_id="${CANARY_RUN_ID:-default}"
  default_root="$experiment_root/canary/$run_id/logs"
  if [[ -z "${CANARY_LOG_DIR:-}" ]]; then
    CANARY_LOG_DIR="${CANARY_LOG_ROOT:-$default_root}/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  fi
  mkdir -p "$CANARY_LOG_DIR"
  export CANARY_LOG_DIR
}

canary_stage_run() {
  local index="$1"
  local total="$2"
  local title="$3"
  local task="$4"
  local log_name="$5"
  local work="$6"
  local summary="$7"
  local log_path="$CANARY_LOG_DIR/$log_name.log"
  local started=$SECONDS
  local next_heartbeat=30
  local pid status elapsed

  printf '\n[%s/%s] %s\n' "$index" "$total" "$title"
  printf '  Task: %s\n' "$task"
  printf '  Log: %s\n' "$log_path"

  (
    set -euo pipefail
    "$work"
  ) >"$log_path" 2>&1 &
  pid=$!
  trap 'kill -TERM '"$pid"' 2>/dev/null || true; wait '"$pid"' 2>/dev/null || true; exit 130' INT TERM

  while kill -0 "$pid" 2>/dev/null; do
    sleep 1 || true
    elapsed=$((SECONDS - started))
    if ((elapsed >= next_heartbeat)); then
      printf '  Status: running (%ss elapsed)\n' "$elapsed"
      next_heartbeat=$((next_heartbeat + 30))
    fi
  done

  if wait "$pid"; then
    status=0
  else
    status=$?
  fi
  trap - INT TERM
  elapsed=$((SECONDS - started))

  if ((status != 0)); then
    printf '  Result: FAIL (exit %s, %ss)\n' "$status" "$elapsed" >&2
    printf '  Last log lines:\n' >&2
    tail -60 "$log_path" >&2 || true
    return "$status"
  fi

  "$summary"
  printf '  Result: PASS (%ss)\n' "$elapsed"
}
