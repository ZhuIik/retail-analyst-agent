# VPS deployment

This is a single-instance demo deployment. The production GCP design remains in
[DESIGN.md](DESIGN.md). Server provisioning and the first deployment are separate steps;
the workflow does not modify an unknown VPS automatically.

## Server prerequisites

For a fresh Ubuntu 24.04 host, an administrator can review and run
[bootstrap-ubuntu.sh](../deploy/bootstrap-ubuntu.sh):

```bash
# Run as root; substitute the existing deployment user.
bash bootstrap-ubuntu.sh DEPLOY_USER
```

It installs Docker Engine and Compose from the
[official Docker apt repository](https://docs.docker.com/engine/install/ubuntu/), prepares
`/opt/retail-analyst`, and adds the deployment user to the Docker group (host-administrator-equivalent
access). Reconnect SSH afterward. It refuses conflicting existing container runtimes rather than
removing them, does not install a reverse proxy or publish the application, and does not explicitly
restart unrelated services. Installing Docker still changes the host's packages/networking;
the administrator should review it before execution on a shared server.

Use a Linux VPS with Docker Engine, the Docker Compose v2 plugin supporting `up --wait`, Bash,
`flock`, and enough memory for the configured 1 GB container limit plus the host/reverse proxy.
Keep `/opt/retail-analyst` owned by the deployment user. That user needs Docker access
(equivalent to host administration); use a dedicated SSH key for GitHub Actions.

Copy `deploy/compose.yaml` and `deploy/deploy.sh` to `/opt/retail-analyst/`. Put a server-specific
`.env` there, based on `.env.example`, with:

- `GOOGLE_API_KEY`: the owner's Gemini key, used by default for hosted demo sessions.
- `GCP_PROJECT_ID` for this installation's BigQuery jobs.
- A private random `ID_SALT`, kept stable between deployments.
- `ANALYST_API_KEYS`: JSON mapping of known user ids to different random keys (32+ characters).
- Optional Langfuse credentials; leave them empty for local-only traces.

Generate random values separately with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Give each reviewer only their user's bearer key. This is a demo authentication mechanism;
the production design uses verified identity-provider JWTs.
Reviewers can optionally use their own Gemini key via `X-Gemini-Api-Key` when creating a
session; see [API.md](API.md). This does not change the server default or BigQuery credentials.
For the public-data prototype, set `ANALYST_DEMO_ENABLED=true` to let reviewers select a demo
persona without entering a key. The daily shared cap defaults to 50 and can be changed with
`ANALYST_DEMO_DAILY_TURNS`. It persists in the reports volume. This is simulated login; do not
enable it for private production data. Static private keys remain available for administration.

Put Google ADC credentials in `/opt/retail-analyst/secrets/google-adc.json`.
Use a dedicated Google identity with BigQuery job creation in the billing project and read
access to the public dataset; do not grant BigQuery write/admin roles. If using service-account
JSON, rotate it and keep it out of Git, image layers and Actions secrets. Federation is preferable
for long-lived production installations. The demo mounts the file read-only.

```bash
cd /opt/retail-analyst
chmod 600 .env
chmod 700 secrets
# The container runs as uid/gid 10001, so it must be able to read this file.
sudo chown 10001:10001 secrets/google-adc.json
sudo chmod 400 secrets/google-adc.json
```

Ensure the deployment user can still check that the file exists. If updating credentials,
replace the file, restore its ownership/permissions, then recreate the container.
The image creates `data` and `traces` owned by uid 10001; fresh named volumes inherit those
permissions. Existing volumes may need ownership repair before an upgrade.

The container binds only to `127.0.0.1:8000`. For private testing use an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 USER@VPS
```

For a public endpoint, put your existing reverse proxy in front of it with HTTPS and a valid
certificate for your domain. Set its upstream timeout to at least 300 seconds for the demo,
limit request body size to 64 KB and apply request rate limits. Never send bearer keys over
public plain HTTP. Do not log authorization headers or request/response bodies. Expose only
the SSH and proxy ports through the firewall; leave 8000 on loopback.

### Optional public HTTPS for the hosted CLI

The repository also includes `compose.public.yaml` and a Caddyfile for a public endpoint.
Enable this only when the server owner has authorized publishing the API on ports 80/443.
Copy both files into the deployment directory, then add to its `.env`:

```dotenv
COMPOSE_FILE=compose.yaml:compose.public.yaml
PUBLIC_ADDRESS=YOUR_PUBLIC_IP_OR_HOSTNAME
```

Run `docker compose up -d gateway` with `ANALYST_IMAGE` set to the installed application image.
The pinned Caddy image obtains and automatically renews a Let's Encrypt certificate with the
`shortlived` profile. Public IP certificates require this profile; inbound ACME validation on
port 80/443 must work and the `caddy_data` volume must persist. See
[Let's Encrypt IP certificates](https://letsencrypt.org/2026/01/15/6day-and-ip-general-availability)
and [Caddy TLS configuration](https://caddyserver.com/docs/caddyfile/directives/tls).
Do not replace the persistent certificate volume on each deployment.

Reviewers can then run `python client.py --server https://YOUR_PUBLIC_IP_OR_HOSTNAME` with a
separately supplied access token. No SSH key or Google credentials need to be shared with them.
The gateway leaves API authorization in place and does not enable access/body logging.
It caps request bodies at 64 KB; the application limits concurrent analysis to one turn.
For a broader public audience, add host/proxy rate limits and per-user daily budgets first.

## GitHub Actions configuration

CI runs on pull requests and pushes to `main`. It checks Ruff, offline tests and consistency
between `uv.lock` and `requirements.txt`, builds the Docker image and boots it without Google
credentials to check `/healthz`. Main then publishes the tested image to GHCR using the
repository's `GITHUB_TOKEN`. No runtime Google secrets are sent through CI.

In the repository's `production` environment configure:

| Kind | Name | Value |
|---|---|---|
| Variable | `VPS_HOST` | VPS IPv4 address or DNS hostname |
| Variable | `VPS_USER` | Dedicated deployment user |
| Variable | `VPS_PORT` | SSH port; defaults to `22` |
| Secret | `VPS_SSH_KEY` | Dedicated private SSH key |
| Secret | `VPS_KNOWN_HOSTS` | Trusted host-key entry, verified against the server/provider console |

Set **repository variable** `VPS_DEPLOY_ENABLED=true` only after the server is ready. It is
checked before the deployment job starts. Leave it unset to build/publish without deploying.
Do not collect and trust a fresh host key inside CI. For a custom port the known-hosts name
uses `[host]:port`. Install the public half of the deployment key in the server user's
`authorized_keys`.

GHCR packages can initially be private. Either make this application's package public (the
image contains only public source/configuration), or log the VPS into GHCR once with a
dedicated credential restricted to pulling packages. Never put that credential in the image.
Allow Actions to write packages if organization/repository policy currently blocks it.

The supplied Actions workflow handles private GHCR packages without a permanent registry
credential on the VPS: its deployment job has `packages: read` and pipes the short-lived
`GITHUB_TOKEN` to `registry-deploy.sh` over SSH. That wrapper creates a private temporary
`DOCKER_CONFIG`, logs in, deploys the digest and removes the temporary registry credentials
on exit. No token is placed in command arguments. Manual pulls of private images still need
registry access; rolling back an already cached image can use `--local-image sha256:...`.

The workflow copies the two deployment files into the existing directory, pulls the immutable
`ghcr.io/...@sha256:...` reference, then runs Compose with `--wait`. Failed health checks restore
the last successfully recorded image. The `.deployed-image` file changes only after success.
Deployments are serialized both in Actions and with a lock on the server.

## First deployment and verification

After CI has published the image, get its digest from GHCR / the image job, then:

```bash
cd /opt/retail-analyst
bash deploy.sh ghcr.io/zhuiik/retail-analyst-agent@sha256:REPLACE_WITH_REAL_64_HEX_DIGEST
curl --fail http://127.0.0.1:8000/healthz
```

Use `/docs` through the tunnel or HTTPS proxy, authorize with Anna's key, and call `/v1/chat`.
Verify a real BigQuery-backed answer, a follow-up with `session_id`, an out-of-scope refusal,
and cancellation/confirmation on a disposable report. `/healthz` checks process/configuration
liveness, not Google credentials, quota or model quality. A healthy deployment therefore still
needs this real smoke test. Offline CI does not certify Google access from the VPS region.

## Rollback and operational limits

Run `bash deploy.sh PREVIOUS_IMAGE_DIGEST` to roll back manually. Do not use a mutable `latest`
tag. Automatic rollback covers the application image with the current Compose configuration;
it does not reverse database migrations or host/proxy changes. The first deployment has no
previous image to restore. The current app has no schema migrations.

Reports and local traces persist in Docker named volumes. Do not run `docker compose down -v`
unless you intend to destroy them. Back up the reports database consistently (SQLite backup
API or while the service is stopped); test restores. Trace files are bounded per string, not
by total disk use: configure disk monitoring and host retention, e.g. remove completed trace
files after 30 days. Backups may retain deleted reports, so define a separate backup expiry
policy; application hard delete is not a promise of instantaneous erasure from backups.

One Uvicorn worker serves at most one analysis turn at a time; other chat requests get 429 with
`Retry-After`. Health checks remain available. Conversations expire after one idle hour, with
a maximum of 128 active sessions. Restart/update/rollback clears all in-memory checkpoints,
including pending confirmations; old session ids return 404 and must not be replayed as consent.
Reports and audit rows persist. For multiple instances or durable conversations, move state
to Postgres and implement distributed concurrency/quotas first.

CLI and HTTP turns run in separate supervised processes with a 120-second execution budget,
including worker startup, session initialization, model calls, retries, queries and trace export.
The supervisor terminates an over-budget worker and escalates to a kill if needed (up to 0.4 s
of process cleanup, plus OS scheduling). A timed-out worker cannot later modify checkpoints or
reports. Its pending confirmation and conversation snapshot are discarded; the user receives
an honest stop message and can start again. User thinking time during deletion confirmation is
not charged; resuming gets a fresh execution budget. Completed checkpoints are transferred
between workers in private temporary files; SDK clients and API keys are not stored in them.
Local JSONL and checkpoint serialization complete before a bounded, best-effort SaaS trace
flush (at most 0.5 seconds within the same deadline). SDK shutdown hooks cannot keep a finished
worker alive; the local trace remains authoritative if an export is interrupted.

An operation committed before the process was stopped is not rolled back. The timeout message
therefore asks the user to check `/reports` before repeating a save/delete. Already accepted
cloud requests may complete remotely; BigQuery receives a job timeout capped to the remaining
turn budget and attempts cancellation on a polling timeout, but server-side cancellation is
best effort. The hard boundary stops local execution, not charges already incurred remotely.

For a manual deployment without GHCR, the same deploy script accepts an existing immutable
Docker image id: `bash deploy.sh --local-image sha256:...`. It never pulls mutable tags in this
mode. CI runs `tests/check_deploy.sh` against the built image: install a healthy image, attempt
a deliberately broken image, verify a nonzero result and automatic restoration of the previous
image. The check uses a separate Compose project, disposable volumes, dummy credentials and
loopback port 18082; no Google calls are made. `DEPLOY_WAIT_TIMEOUT` defaults to 180 seconds and
can be shortened for this integration check. Optional Caddy configuration is validated before
application replacement and reloaded after success.

References: [GitHub's container publishing guide](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images),
[FastAPI deployment concepts](https://fastapi.tiangolo.com/deployment/concepts/).
