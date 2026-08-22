# Cloud deployment — Railway + Postgres

Hosts the Pupa FastAPI backend on Railway with a managed Postgres,
Langfuse Cloud tracing, and multi-tenant safe defaults (shell off,
MCP servers off). Pair an iOS device against the hosted backend and you're
in business.

Runtime config layout in the cloud:

| Source | What it sets | Where |
|---|---|---|
| `deploy/cloud-config.yml` (baked into image) | Same schema as a local `~/.pupa-backend/config.yml`. Lists both supported providers under `llm_providers` with `default_llm_provider: anthropic`. Pins the multi-tenant safety posture (`shell_tool_enabled: false`, `screenshare: false`, no `mcp_servers` block). Tracing needs no key here — it is opt-out, active as soon as the Langfuse credentials are present in the environment. | `/root/.pupa-backend/config.yml` inside container |
| Railway env vars | Secrets (LLM creds, `PUPA_API_KEY`, Langfuse keys, optional Tavily) and `DATABASE_URL` (auto-injected by the Postgres plugin). Shell env overrides YAML — see [`backend/pupa_backend/pupa_config.py`](../backend/pupa_backend/pupa_config.py). | Railway UI → Variables |

## One-time setup

### 1. Langfuse Cloud

1. Sign up at https://cloud.langfuse.com (free tier).
2. Create a project; copy `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.

### 2. Railway project

1. https://railway.com → New Project → Deploy from GitHub repo → select this repo.
2. Railway auto-detects [`railway.json`](../railway.json) + [`Dockerfile`](../Dockerfile). First deploy will fail until env vars are set (next step) — that's expected.
3. Add the **Postgres** plugin (right-hand panel → New → Database → PostgreSQL). Railway injects `DATABASE_URL=postgres://...` into the service automatically; [`backend/pupa_backend/db_config.py`](../backend/pupa_backend/db_config.py) binds both the checkpointer and the store to it.

### 3. Required service env vars

Set these in Railway → service → Variables:

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic creds. Required for the default provider (`anthropic`, set by the baked YAML). |
| `PUPA_API_KEY` | Random 32-byte hex string — bootstrap pairing credential. `openssl rand -hex 32`. Can be removed after the first device is paired. |

**Required, from Railway Postgres:**

| Var | Purpose |
|---|---|
| `DATABASE_URL` | Read by [`backend/pupa_backend/db_config.py`](../backend/pupa_backend/db_config.py); its scheme selects the backend for both the checkpointer and the store. The baked YAML pins `persistence.require_db_scheme: postgresql`, so a missing or non-Postgres URL fails the lifespan loudly instead of silently falling back to in-memory (chat history wiped every restart). On Railway, add `DATABASE_URL` to the backend service's Variables as a reference: `${{Postgres.DATABASE_URL}}`. |

**Optional:**

| Var | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | Langfuse Cloud keys from step 1. The tracer no-ops without them. |
| `TAVILY_API_KEY` | Enables the `tavily_search` backend tool. |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_DEFAULT_REGION` | Unlocks the Bedrock models in the picker (see [full menu](#enabling-the-full-model-menu)). Bedrock entries use EU cross-region inference profiles, so region `eu-west-1`. |
| `OPENROUTER_API_KEY` | Unlocks the OpenRouter models in the picker (GLM, Qwen, MiniMax, Kimi, DeepSeek). Single key for all of them — get one at https://openrouter.ai/keys. |

#### Enabling the full model menu

The iOS picker fetches `GET /harnesses`, whose deepagents entry lists **every** model in
`MODEL_REGISTRY` ([`backend/pupa_backend/harnesses/langgraph/agent.py`](../backend/pupa_backend/harnesses/langgraph/agent.py)) — Anthropic,
Bedrock, and OpenRouter. The client sends its chosen `(provider, modelId)`
pair per request; the backend builds that model on the fly. This selection is
**independent of `LLM_PROVIDER`** — `LLM_PROVIDER` only picks the default model
used when a client sends no choice.

That means provider creds are **additive, not mutually exclusive**. To make the
whole menu actually usable on Railway (rather than 401/failing when a user picks
a model whose creds are absent), set every provider's key alongside the default:

| Provider | Creds to set on Railway |
|---|---|
| Anthropic (default) | `ANTHROPIC_API_KEY` |
| Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_DEFAULT_REGION=eu-west-1` |
| OpenRouter | `OPENROUTER_API_KEY` |

You do **not** change `LLM_PROVIDER` to add a provider — leave it unset (default
`anthropic` from the baked YAML) and just supply the keys. A model whose creds
are missing surfaces a `MissingCredentialsError` to the client as an AG-UI error
toast, so set all three keys to avoid dead entries in the picker.

**Do NOT set** `SHELL_TOOL_ENABLED`, `PUPA_SCREENSHARE`, or `PUPA_MCP_SERVERS` on Railway — the baked-in `cloud-config.yml` keeps shell/screenshare off and ships no MCP servers. Setting them via shell env would override that safety default in a multi-tenant deployment.

#### Changing the default provider

The above unlocks models for **per-request** selection. To change the *default*
model (used when a client sends no choice), set `LLM_PROVIDER`. The baked YAML
lists both `anthropic` and `bedrock` under `llm_providers`; to default to Bedrock
without rebuilding the image, set on Railway:

| Var | Value |
|---|---|
| `LLM_PROVIDER` | `bedrock` |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_DEFAULT_REGION` | Bedrock creds (also unlocks the Bedrock picker entries). |

Shell env wins over YAML, so `LLM_PROVIDER=bedrock` overrides the YAML's `default_llm_provider: anthropic`. For an OpenAI-compatible proxy, set `LLM_PROVIDER=openai_compatible` plus `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

### 4. Pair your first device

After the deploy is green (Railway → service → URL gives you `https://<svc>.up.railway.app`):

```bash
make pair URL=https://<svc>.up.railway.app KEY="$PUPA_API_KEY_RAILWAY" \
          LABEL="My iPhone" CODE_TTL=3600 DEVICE_TTL=432000
```

This prints the 8-char code **and renders a scannable QR** encoding a
`pupa-pair://` deep link with the Railway URL baked in — so the phone doesn't
have to type either. Open iOS Settings → Backend → Edit and scan it (or paste
URL + code by hand).

- `CODE_TTL` — how long the bootstrap code stays redeemable. Seconds, `1..86400`
  (the API rejects anything longer). Default 300.
- `DEVICE_TTL` — lifetime of the *device token* the code mints. Seconds, no cap.
  `432000` = 5 days; `2592000` = 30 days. Omit for a token that never expires —
  fine for your own phone on a LAN backend, not for a shared cloud deploy.
- `KEY` — that backend's `PUPA_API_KEY`. Only needed when it differs from your
  local one, which it does for Railway. Shell env wins over
  `~/.pupa-backend/config.yml`, so this cleanly overrides the local key.

The raw HTTP equivalent, if you'd rather not use `make`:

```bash
curl -X POST \
  -H "Authorization: Bearer $PUPA_API_KEY_RAILWAY" \
  -H "Content-Type: application/json" \
  -d '{"label":"My iPhone","codeTtlSeconds":3600,"deviceTokenTtlSeconds":432000}' \
  https://<svc>.up.railway.app/auth/pair/begin
```

No QR — the response is just JSON with the `code`.

### 5. Keep the bootstrap key

`PUPA_API_KEY` is the **only** credential `/auth/pair/begin` accepts — a paired-device token gets 403, so devices can't mint devices. Delete the key and you can't pair anything else without setting it again. Keep it in Railway's variables (never in a client), and rotate it rather than removing it.

## Transport security — required for anything reachable

**Any internet-reachable deploy must terminate TLS and set
`PUPA_REQUIRE_HTTPS=1`** (config: `transport.require_https: true`). It's
already pinned on in [`deploy/cloud-config.yml`](../deploy/cloud-config.yml),
so Railway inherits it.

The reason is the pairing handshake: `/auth/pair` returns the device token in
plaintext exactly once. Over a plaintext hop, anyone on the path has a
credential that works until it's revoked. Every request after that carries a
bearer token, so the exposure isn't limited to pairing.

TLS itself stays optional because `pupa-backend` is self-hosted first — an
offline or LAN install has no name to put a cert on. The flag is how a
reachable deploy opts into strictness:

| Deploy | TLS terminated by | `PUPA_REQUIRE_HTTPS` |
|---|---|---|
| Railway | Railway's edge (`X-Forwarded-Proto: https`) | **1** (pinned in the image config) |
| Cloudflare tunnel | `cloudflared` | **1** |
| Tailscale serve | `tailscale serve` with tailnet HTTPS | **1** |
| Own cert | the backend (`PUPA_TLS_CERT` / `PUPA_TLS_KEY`) | **1** |
| LAN / offline self-host | nothing | unset — plaintext on the local segment |

Behind any reverse proxy other than the Tailscale/Cloudflare modes the backend
starts itself — **including Railway** — also set `PUPA_TRUSTED_PROXY=1` (config
`transport.trusted_proxy`, already pinned in the cloud image). TLS terminates at
the proxy, so `X-Forwarded-Proto` is the only evidence the caller's hop was
encrypted, and the backend ignores that header unless it's told something in
front actually writes it. Leave it **off** on a direct bind: there the header is
written by the caller, and believing it would let anyone assert their own
plaintext hop was TLS.

When set, a non-TLS request gets `403 HTTPS required` and the screen-share
socket closes with 4403. Health probes are exempt so platform checks still
pass. There is **no** loopback exemption: every tunnel mode terminates in
front of a loopback-bound listener, so "it came from 127.0.0.1" is true of
every remote caller. For local development, leave the flag unset rather than
looking for a carve-out — `http://localhost:8004` then works exactly as before.

## Auto-deploy on `main`

Railway's native GitHub integration redeploys on every push to `main`. Per [CONTRIBUTING.md](../CONTRIBUTING.md), `main` only fast-forwards from `dev` at release time, so deploys match the release cadence — no GitHub Action needed.

## Verifying locally first

The same image runs on a laptop:

```bash
docker build -t pupa-backend .
docker run --rm -p 8004:8004 \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e PUPA_API_KEY=test \
  -e DATABASE_URL=postgresql://user:pass@host.docker.internal:5432/pupa \
  pupa-backend
```

Then `curl http://localhost:8004/auth/config` → `{"authRequired": true, "methods": ["api_key"], ...}`. Startup logs should show `persistence ready: checkpointer=AsyncPostgresSaver, store=AsyncPostgresStore`.
