#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: scripts/run-harness.sh <config> <adapter-module> [socket]}"
ADAPTER_MODULE="${2:?usage: scripts/run-harness.sh <config> <adapter-module> [socket]}"
SOCKET="${3:-}"
CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$REPO/$CONFIG_PATH"
ARGS=(serve --config "$CONFIG_PATH" --adapter-module "$ADAPTER_MODULE")
[[ -n "$SOCKET" ]] && ARGS+=(--socket "$SOCKET")
exec uv run harness "${ARGS[@]}"
