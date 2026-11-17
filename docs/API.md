# HTTP API

For an interviewer who only wants to chat in the terminal, run `python client.py` from the
repository (Python 3.11+, no dependency installation). Choose Anna, Mark or CEO from its menu,
or use `python client.py --user anna`. No access key is entered manually for the public demo.
It uses the hosted HTTPS endpoint by default. `--own-gemini-key` adds a hidden prompt for a personal
Gemini key. The remaining instructions describe direct HTTP access or a local server.

Set `ANALYST_API_KEYS` in `.env` to a JSON mapping of user ids from [config/users.yaml](../config/users.yaml) to
separate random keys of at least 32 characters; also set a private `ID_SALT`.
Generate each value with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
The API derives the user and brand scope from the key. Clients cannot choose another user.
This static-key mode remains available for private access and administration.

## Public demo user selection

Set `ANALYST_DEMO_ENABLED=true` to expose `GET /v1/demo/users` and `POST /v1/demo/access`.
The latter accepts `{"user":"anna"}` (also `mark` or `ceo`) and issues temporary access for
one hour. The CLI performs this exchange automatically and holds the token only in memory.
The public choice of persona is simulated login, not real identity verification. Anyone can
select the CEO's scope over the public dataset. Keep this mode disabled for private data.

Each visitor receives a separate internal owner id. Choosing Anna again cannot open a previous
visitor's reports, traces or conversation. Client restarts create a new visit and report library.
Static-key users' existing reports are also isolated from demo visitors.

`ANALYST_DEMO_DAILY_TURNS` defaults to 50 total chat requests per UTC day across all demo visitors,
including failed attempts, confirmations and personal-Gemini-key sessions. The counter is stored
in `data/demo-budget.sqlite` beside the reports database and survives process restarts. It is
charged before session construction or any model/BigQuery operation. At the limit the API returns
429. Demo access issuance is limited to 128 active visitors. Private static-key access retains
the ordinary per-turn limits and does not consume this public-demo quota.

```bash
uv run uvicorn retail_analyst.api:app --host 127.0.0.1 --port 8000 --workers 1
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $ANNA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"message":"Compare revenue for my brands over the last full month."}'
```

`ANNA_API_KEY` in the curl example is the same value configured for Anna on the server.
The response contains `session_id`, `trace_id`, `answer` and `confirmation`. Include the returned
`session_id` in the next request to continue. If `confirmation` is present, show its exact preview
to the user and send their reply as the next `message` in that session. Never auto-confirm.
An invalid key returns 401; another user's, expired or restarted session returns 404.
The demo serves one turn at a time (429 with `Retry-After` while busy), at most 128 sessions,
with a one-hour idle expiry. Restarting the process clears conversations and pending confirmations.
Saved reports and local JSONL files survive when deployed with the supplied volumes.
Interactive API documentation: `/docs` (use Authorize to enter the user's key).

## Gemini key: server default or your own

By default, sessions use the `GOOGLE_API_KEY` configured on the server. Reviewers can try the
hosted demo without obtaining a Gemini key. To use a personal Gemini key, supply the optional
`X-Gemini-Api-Key` header on the **first** `POST /v1/chat` request (without `session_id`).
The same field is available under Parameters in `/docs`.

```bash
curl https://YOUR_DOMAIN/v1/chat \
  -H "Authorization: Bearer $ANNA_API_KEY" \
  -H "X-Gemini-Api-Key: $MY_GEMINI_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Compare revenue for my brands last month."}'
```

Continue with the returned `session_id`, **omitting** `X-Gemini-Api-Key`. The selected key stays
with that session, including model fallbacks and deletion confirmation. To change it or return
to the server default, start a new session. An invalid personal key does not silently fall back
to the server owner's key. Other sessions continue to use their own selected key.

The personal key is held in process memory for the session lifetime, is not part of graph
messages or responses, and is filtered from local JSONL errors if an SDK error repeats it.
Do not put it in the question text or URL. Send it only over HTTPS or a local SSH tunnel.
The bearer key still controls identity and brand access; the personal Gemini key only selects
the model credentials/quota. BigQuery continues to use the server's Google Cloud identity.

For a local installation, set your own `GOOGLE_API_KEY` in `.env` instead.

Deployment and CI/CD configuration: [DEPLOYMENT.md](DEPLOYMENT.md).

The hosted CLI also uses owner-scoped `GET /v1/reports` and
`GET /v1/sessions/{session_id}/trace`. Both require the same bearer token; expired/foreign
session traces return 404. Personal Gemini keys are never sent for these read operations.
