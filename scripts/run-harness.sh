#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: scripts/run-harness.sh <config>}"
CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$REPO/$CONFIG_PATH"
exec uv run harness serve --config "$CONFIG_PATH"
