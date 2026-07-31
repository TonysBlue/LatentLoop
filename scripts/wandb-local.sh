#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-status}"
image="wandb/local:0.83.0@sha256:b234c9d084b65164598da6fa5f17d38ec71137b037c059b46889c1495b008c52"
container="latentloop-wandb"
volume="latentloop-wandb"

case "$command_name" in
  up)
    docker volume create "$volume" >/dev/null
    if docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
      docker start "$container" >/dev/null
    else
      docker run -d \
        --name "$container" \
        --restart unless-stopped \
        --cpus 2 \
        --memory 4g \
        -p 127.0.0.1:8080:8080 \
        -v "$volume:/vol" \
        "$image" >/dev/null
    fi
    for _ in $(seq 1 60); do
      if curl --fail --silent http://127.0.0.1:8080/ready >/dev/null; then
        printf 'W&B Local is ready at http://127.0.0.1:8080\n'
        exit 0
      fi
      sleep 2
    done
    docker logs --tail=100 "$container" >&2
    printf 'W&B Local did not become healthy within 120 seconds\n' >&2
    exit 1
    ;;
  down)
    docker stop "$container" >/dev/null 2>&1 || true
    docker rm "$container" >/dev/null 2>&1 || true
    ;;
  status)
    docker ps -a --filter "name=^/${container}$"
    ;;
  logs)
    docker logs --tail=200 "$container"
    ;;
  *)
    printf 'usage: %s {up|down|status|logs}\n' "$0" >&2
    exit 2
    ;;
esac
