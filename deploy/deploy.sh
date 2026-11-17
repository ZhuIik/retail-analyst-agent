#!/usr/bin/env bash
# Run from the deployment directory. Requires Docker Compose v2 with --wait support.
set -euo pipefail
cd -- "$(dirname -- "$0")"
local_image=false
if [[ "${1:-}" == --local-image ]]; then
  local_image=true
  shift
fi
image="${1:?Usage: deploy.sh [--local-image] IMMUTABLE_IMAGE}"
if "$local_image"; then
  [[ "$image" =~ ^sha256:[a-f0-9]{64}$ ]] || { echo 'Expected a local immutable image id' >&2; exit 2; }
  docker image inspect "$image" >/dev/null
else
  [[ "$image" =~ ^ghcr\.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$ ]] || {
    echo 'Expected an immutable GHCR sha256 image reference' >&2; exit 2;
  }
fi
wait_timeout="${DEPLOY_WAIT_TIMEOUT:-180}"
[[ "$wait_timeout" =~ ^[0-9]+$ ]] && ((wait_timeout >= 1 && wait_timeout <= 600)) || exit 2
test -f .env
test -f secrets/google-adc.json
# Also serialize manual deployments on the VPS.
exec 9>.deploy.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }
previous=''
if [[ -f .deployed-image ]]; then previous="$(cat .deployed-image)"; fi
export ANALYST_IMAGE="$image"
if ! "$local_image"; then docker compose pull; fi
# Fail before replacing the application if an optional public gateway config is invalid.
gateway=false
if docker compose config --services | grep -qx gateway; then
  gateway=true
  docker compose run --rm --no-deps gateway caddy validate --config /etc/caddy/Caddyfile
fi
if docker compose up -d --pull never --wait --wait-timeout "$wait_timeout"; then
  if "$gateway"; then docker compose exec -T gateway caddy reload --config /etc/caddy/Caddyfile; fi
  printf '%s\n' "$image" > .deployed-image.next
  mv -- .deployed-image.next .deployed-image
  echo 'Deployment healthy. Conversations from the previous process must be restarted.'
else
  echo 'New image failed its health check.' >&2
  if [[ -n "$previous" ]]; then
    export ANALYST_IMAGE="$previous"
    docker compose up -d --pull never --wait --wait-timeout "$wait_timeout"
    echo 'Restored previous image.' >&2
  fi
  exit 1
fi
