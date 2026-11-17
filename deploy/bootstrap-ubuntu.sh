#!/usr/bin/env bash
# One-time administrator step, using Docker's official Ubuntu apt repository.
# Usage as root: bash bootstrap-ubuntu.sh opsfleet
set -euo pipefail
[[ "$EUID" -eq 0 ]] || { echo 'Run this script as root.' >&2; exit 1; }
deploy_user="${1:?Specify the deployment user}"
[[ "$deploy_user" =~ ^[a-z_][a-z0-9_-]*$ ]] || exit 2
id "$deploy_user" >/dev/null
. /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || {
  echo 'This bootstrap supports Ubuntu 24.04 only.' >&2; exit 1;
}

if ! command -v docker >/dev/null; then
  # Never remove/replace existing runtimes on a shared host automatically.
  for package in docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc; do
    if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
      echo "Existing package $package requires administrator review before installing Docker CE." >&2
      exit 1
    fi
  done
  export DEBIAN_FRONTEND=noninteractive
  # Prevent needrestart from automatically restarting unrelated host services.
  export NEEDRESTART_MODE=l
  apt-get update
  apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl --fail --silent --show-error https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod 0644 /etc/apt/keyrings/docker.asc
  cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker
docker compose version
# Docker group membership grants host-administrator-equivalent access.
usermod -aG docker "$deploy_user"
target=/opt/retail-analyst
[[ ! -L "$target" ]] || { echo 'Deployment path must not be a symlink.' >&2; exit 1; }
install -d -m 0750 -o "$deploy_user" -g "$(id -gn "$deploy_user")" "$target"
install -d -m 0700 -o "$deploy_user" -g "$(id -gn "$deploy_user")" "$target/secrets"
echo 'Bootstrap complete. Reconnect SSH to pick up Docker group membership.'
