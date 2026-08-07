#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE=""
RUN_ID=""
OVERRIDES=()
while (($#)); do
  case "$1" in
    --recipe) RECIPE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --set) OVERRIDES+=(--set "$2"); shift 2 ;;
  *) printf 'usage: %s --recipe configs/recipes/<name>.yaml [--run-id ID]\n' "$0" >&2; exit 2 ;;
  esac
done
[[ -n "$RECIPE" ]] || { printf '%s\n' '--recipe is required' >&2; exit 2; }
cd "$REPO"
ARGS=(--recipe "$RECIPE")
[[ -n "$RUN_ID" ]] && ARGS+=(--run-id "$RUN_ID")
exec uv run latentloop run-recipe "${ARGS[@]}" "${OVERRIDES[@]}"
