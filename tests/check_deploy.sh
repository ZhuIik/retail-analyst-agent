#!/usr/bin/env bash
# Real Docker integration: good image -> failing image -> automatic restore. No Google access.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
image="${1:?Pass the tested application image name or id}"
good="$(docker image inspect --format '{{.Id}}' "$image")"
workdir="$(mktemp -d /tmp/retail-deploy-check.XXXXXXXX)"
export COMPOSE_PROJECT_NAME="retail-deploy-check-$$"
unset COMPOSE_FILE
export DEPLOY_WAIT_TIMEOUT=20
export ANALYST_IMAGE="$good"
bad=''
source_container=''
cleanup() {
  (cd -- "$workdir" && docker compose down -v --remove-orphans) || true
  if [[ -n "$source_container" ]]; then docker rm "$source_container" >/dev/null || true; fi
  if [[ -n "$bad" ]]; then docker image rm "$bad" >/dev/null || true; fi
  case "$workdir" in /tmp/retail-deploy-check.*) rm -rf -- "$workdir" ;; esac
}
trap cleanup EXIT
cp "$repo/deploy/deploy.sh" "$workdir/deploy.sh"
sed 's/127.0.0.1:8000:8000/127.0.0.1:18082:8000/' "$repo/deploy/compose.yaml" > "$workdir/compose.yaml"
mkdir -m 700 "$workdir/secrets"
printf '{}\n' > "$workdir/secrets/google-adc.json"
cat > "$workdir/.env" <<'EOF'
ANALYST_API_KEYS={"anna":"ci-only-access-key-with-at-least-32-characters"}
ID_SALT=ci-only-private-id-salt
ANALYST_DEMO_ENABLED=false
EOF
chmod 600 "$workdir/.env"
bash "$workdir/deploy.sh" --local-image "$good"
[[ "$(cat "$workdir/.deployed-image")" == "$good" ]]
curl --fail --silent http://127.0.0.1:18082/healthz
source_container="$(docker create "$good")"
bad="$(docker commit --change 'CMD ["python", "-c", "raise SystemExit(1)"]' "$source_container")"
if bash "$workdir/deploy.sh" --local-image "$bad"; then
  echo 'Broken deployment incorrectly reported success' >&2
  exit 1
fi
[[ "$(cat "$workdir/.deployed-image")" == "$good" ]]
curl --fail --silent http://127.0.0.1:18082/healthz
actual="$(cd -- "$workdir" && docker compose images -q api)"
[[ "$(docker image inspect --format '{{.Id}}' "$actual")" == "$good" ]]
echo 'Deployment integration passed: healthy image restored automatically.'
