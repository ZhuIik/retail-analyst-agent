#!/usr/bin/env bash
# GitHub's short-lived package-read token arrives only on stdin, never in command arguments.
set -euo pipefail
cd -- "$(dirname -- "$0")"
actor="${1:?Specify the GitHub actor}"
image="${2:?Specify the immutable GHCR image}"
[[ "$image" =~ ^ghcr\.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$ ]] || exit 2
registry_config="$(mktemp -d /tmp/retail-ghcr.XXXXXXXX)"
cleanup() {
  case "$registry_config" in /tmp/retail-ghcr.*) rm -rf -- "$registry_config" ;; esac
}
trap cleanup EXIT
export DOCKER_CONFIG="$registry_config"
docker login ghcr.io --username "$actor" --password-stdin
bash ./deploy.sh "$image"
