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
curl -X POST \
  -H "Authorization: Bearer $PUPA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label":"My iPhone","codeTtlSeconds":600,"deviceTokenTtlSeconds":259200}' \
  https://<svc>.up.railway.app/auth/pair/begin
```

The response includes an 8-char `code`. Open iOS Settings → Backend → Edit → paste the URL and the code. The token expires after 3 days (`259200` seconds) — re-pair when it lapses. For a longer-lived token (e.g. 30 days) pass `"deviceTokenTtlSeconds":2592000`. Omit the field entirely for a non-expiring token.

### 5. (Recommended) Remove the bootstrap key

Once you trust your paired devices, delete `PUPA_API_KEY` from Railway. Paired-device tokens can still mint new codes via `/auth/pair/begin`; fresh laptops with no token get 401 until the next pair.

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
