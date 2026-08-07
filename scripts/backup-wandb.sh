#!/usr/bin/env bash
set -euo pipefail

backup_root="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}/backups/wandb"
mkdir -p "$backup_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
target="$backup_root/wandb-local-$timestamp.tar.gz"

was_running="$(docker inspect -f '{{.State.Running}}' latentloop-wandb 2>/dev/null || true)"
if [[ "$was_running" == "true" ]]; then
  docker stop latentloop-wandb >/dev/null
  trap 'docker start latentloop-wandb >/dev/null' EXIT
fi
docker run --rm \
  -v latentloop-wandb:/source:ro \
  -v "$backup_root:/backup" \
  alpine:3.22 \
  tar -C /source -czf "/backup/$(basename "$target")" .

find "$backup_root" -maxdepth 1 -name 'wandb-local-*.tar.gz' -printf '%T@ %p\n' \
  | sort -nr \
  | awk 'NR > 4 {print $2}' \
  | xargs -r rm -f

if [[ "$was_running" == "true" ]]; then
  docker start latentloop-wandb >/dev/null
  trap - EXIT
  for _ in $(seq 1 120); do
    if curl --fail --silent http://127.0.0.1:8080/ready >/dev/null; then
      break
    fi
    sleep 1
  done
  if ! curl --fail --silent http://127.0.0.1:8080/ready >/dev/null; then
    printf 'W&B Local did not recover after backup\n' >&2
    exit 1
  fi
fi

printf '%s\n' "$target"
