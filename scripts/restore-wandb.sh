#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s BACKUP.tar.gz\n' "$0" >&2
  exit 2
fi

backup="$(realpath "$1")"
if [[ ! -f "$backup" ]]; then
  printf 'backup not found: %s\n' "$backup" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -qx latentloop-wandb; then
  printf 'remove the W&B container with scripts/wandb-local.sh down before restore\n' >&2
  exit 1
fi

docker volume create latentloop-wandb >/dev/null
docker run --rm \
  -v latentloop-wandb:/target \
  -v "$(dirname "$backup"):/backup:ro" \
  alpine:3.22 \
  sh -c 'find /target -mindepth 1 -delete && tar -C /target -xzf "/backup/$1"' \
  sh "$(basename "$backup")"

printf 'restored %s into latentloop-wandb\n' "$backup"
