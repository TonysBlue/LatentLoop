#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$REPO/scripts/run-canary.sh" prepare
"$REPO/scripts/run-canary.sh" encode
