#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: scripts/run-harness.sh <config> [socket]}"
SOCKET="${2:-}"
CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$REPO/$CONFIG_PATH"
ARGS=(serve --config "$CONFIG_PATH")
[[ -n "$SOCKET" ]] && ARGS+=(--socket "$SOCKET")
exec uv run harness "${ARGS[@]}"
