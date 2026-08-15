# Pupa backend 

[![pupa-backend](https://img.shields.io/badge/backend-0.0.87-3776ab?logo=python&logoColor=white)](backend/pyproject.toml)
[![screenshare](https://img.shields.io/badge/screenshare-0.0.6-f05138?logo=swift&logoColor=white)](screenshare-sidecar/Sources/PupaScreenshare/Version.swift)

Spin up your own server so a **local agent drives the
[Pupa](https://pupa-app.com) app**. Run it on your laptop right next to the app —
or in the cloud — pair your phone **once**, and your chosen agent takes it from
there: use **Claude Code** (billed to your Claude subscription) or bring your
own. Reach the backend however suits you — your local network, a **Cloudflare**
tunnel, or **Tailscale**.

Under the hood it's a FastAPI server speaking
[AG-UI](https://github.com/ag-ui-protocol/ag-ui) to the app over one `POST /`
stream, with pluggable [agent harnesses](#agent-harnesses) behind that one
contract. Runs locally or in the cloud (Railway + Postgres).

## Contents

- [Components](#components)
- [Quick start](#quick-start)
- [Agent harnesses](#agent-harnesses)
- [Optional features](#optional-features)
- [Architecture](#architecture)
- [Auth model](#auth-model)
- [Connecting the app](#connecting-the-app)
- [Screen-share sidecar](#screen-share-sidecar)
- [Deploy](#deploy)
- [Contributing](#contributing)
- [Built on](#built-on)

## Components

| Component | Version | Where |
|---|---|---|
| **Backend** (Python · FastAPI · Claude Code + deepagents/LangGraph harnesses) | `0.0.87` | [`backend/pyproject.toml`](backend/pyproject.toml) |
| **Screen-share sidecar** (Swift CLI, macOS-only, optional) | `0.0.6` | [`screenshare-sidecar/Sources/PupaScreenshare/Version.swift`](screenshare-sidecar/Sources/PupaScreenshare/Version.swift) |

Patch-only bumps (`0.0.X` → `0.0.X+1`). See
[CHANGELOG.md](CHANGELOG.md) for the root project version and
[CONTRIBUTING.md](CONTRIBUTING.md) for the release flow.

## Quick start

Install from PyPI with [uv](https://docs.astral.sh/uv/) (isolated tool venv, puts
the `pupa-backend` CLI on your PATH). Pin a version to match your app:

```bash
uv tool install "pupa-backend[setup]"          # latest
uv tool install "pupa-backend[setup]==0.0.72"  # pinned to a known-good app pairing
```

Or the one-line source installer (clones this repo, installs `uv`, runs the
setup wizard, drops the `pupa-backend` CLI in `~/.local/bin/`) — needed if you
also want the macOS screen-share sidecar, which is built from source:

```bash
curl -fsSL https://raw.githubusercontent.com/pupa-app/pupa-backend/main/install.sh | bash
```

After either install:

```bash
pupa-backend run            # start the backend on :8004
pupa-backend pair           # mint a QR pairing code for your iPhone
pupa-backend status         # is it running?
pupa-backend service-install  # run as launchd / systemd background service
```

Or from a clone, using `make`:

```bash
make install         # uv sync of backend deps
make setup           # interactive wizard — writes ~/.pupa-backend/config.yml
make backend         # run on :8004
make pair            # mint pairing code
```

`make help` lists every target. LLM credentials live in your shell env
(see [`.env.example`](.env.example)), never in the config file.

### Fast same-laptop dev loop

`PUPA_AUTH_DISABLED=1` skips pair-once auth entirely — same-machine only,
never on a reachable backend:

```bash
PUPA_AUTH_DISABLED=1 make backend
```

## Agent harnesses

Everything around the agent loop is the same no matter which model answers you —
that's the point of the backend. The AG-UI `POST /` SSE contract, pair-once auth,
and, crucially, **your app's tools** are harness-independent. The client forwards
its tool definitions with each request; the backend hands them to the model, and
when the model calls one it round-trips back to the device to run it (the AG-UI
interrupt/resume contract). The app's wire protocol is identical whichever
harness is in use.

An **agent harness** is the swappable inner loop — the piece that talks to a
model and decides when to call those tools. **Two ship today, and the registry is
pluggable so more can be added.** Every *enabled* harness is mounted at once at
`POST /harnesses/{id}` (the default one is also aliased at `POST /`), and the iOS
app picks which one to use per backend connection.

### Claude Code loop (`claude_code`) — what most people will use

Runs the full Claude Code agent in-process via the **Claude Agent SDK**, which
drives the `claude` CLI. Billing is **your Claude subscription** — the harness
refuses to start if per-token API credentials (`ANTHROPIC_API_KEY`, `AWS_*`, …)
are present, so you're never billed per token. You get Claude's own tools —
`Read` / `Bash` / `Grep`, `web_search`, `web_fetch`, … — rendered live in the
app, *plus* your app's forwarded tools executed on-device. No provider keys and
no model wiring: just install and log into Claude Code (the `claude` CLI).

### Deepagents (`deepagents`)

The graph-based harness — a **deepagents** agent (LangChain `create_agent` +
deepagents middleware) running on **LangGraph**. Billing is **per-token API**
through a provider you configure — AWS Bedrock, Anthropic, or any
OpenAI-compatible endpoint. It brings its own backend-side tools —
`tavily_search`, the local shell tool, subagent delegation (`task`), and agent
skills (see [Optional features](#optional-features)) — plus a per-request model
swap and full LangGraph persistence (checkpointer + store on SQLite / Postgres /
in-memory). Reach for it when you want a specific model or provider, or don't
have a Claude subscription.

### Which harness?

| | **Claude Code loop** (`claude_code`) | **Deepagents (LangGraph)** (`deepagents`) |
|---|---|---|
| Billing | Your Claude subscription | Per-token API |
| Credentials | Logged-in `claude` CLI | Bedrock / Anthropic / OpenAI-compatible key |
| Models | Claude (subscription) | Any provider model, swappable per request |
| Built-in tools | Claude's native + server tools (`Read`, `Bash`, `web_search`, …) | `tavily_search`, shell, subagents, skills |
| MCP servers | ✅ shared connection | ✅ shared connection |
| Your app's tools | forwarded, executed on-device | forwarded, executed on-device |
| Persistence | Claude session resume | LangGraph checkpointer + store |

MCP servers (and Langfuse tracing, and the screen-share broker) are
harness-independent — configure them once and they work with either harness.
Enable and choose harnesses in `config.yml`'s `harnesses:` block (or the
`PUPA_HARNESSES` JSON override). With nothing set, the deepagents (LangGraph)
harness runs alone; enable both and the iOS app picks per connection.

## Optional features

> MCP servers, Langfuse tracing, and the screen-share broker are
> **harness-independent** (they work with either harness). The rest below —
> `tavily_search`, the shell tool, subagents, skills — are backend tools for the
> **[deepagents (LangGraph) harness](#agent-harnesses)**; the Claude Code loop
> brings Claude's own toolset instead.

The home for configuration is `~/.pupa-backend/config.yml` — written by
`make setup` (`pupa-backend setup`) and safe to edit by hand. Set
optional features as keys there; the corresponding env var is an
**override** for containers / CI / one-off runs (shell env wins over
config.yml at load time). The full annotated env reference is
[`.env.example`](.env.example). Langfuse, for instance, is config-only —
add the keys, restart, done; no code changes.

| Feature | `config.yml` | Env var override | Notes |
|---|---|---|---|
| **Web search** (`tavily_search` tool) | `tavily_api_key: tvly-…` | `TAVILY_API_KEY` | A deepagents-harness backend tool. Absent → not registered; agent falls back to training data. Key: [app.tavily.com](https://app.tavily.com). |
| **Langfuse tracing** | `langfuse:`<br>`  public_key: pk-lf-…`<br>`  secret_key: sk-lf-…`<br>`  disabled: true` to turn off | `PUPA_LANGFUSE_DISABLED=1` · `LANGFUSE_PUBLIC_KEY` · `LANGFUSE_SECRET_KEY` | **On by default** whenever credentials are present — auto-traces every AG-UI request. Self-hosted host via `LANGFUSE_BASE_URL` (env only); omit for Langfuse Cloud. |
| **Local shell tool** | `shell_tool_enabled: true` | `SHELL_TOOL_ENABLED` | Agent runs host shell commands; per-command approval on by default. Advanced knobs (`SHELL_TOOL_WORKSPACE`, `SHELL_PASS_ENV`, `SHELL_ENV_EXCLUDE`) are env-only. Trusted dev hosts only. |
| **Screen-share broker** (macOS) | `screenshare: true` | `PUPA_SCREENSHARE` | Mounts the WebRTC broker at `/screenshare/ws`; pair with `pupa-backend screenshare`. No-op on Linux. |
| **MCP servers** (incl. Playwright browser tools) | `mcp_servers:` block — e.g. `pupa-backend mcp add --playwright` | `PUPA_MCP_SERVERS` (JSON) | Any stdio/HTTP MCP server, connected once and shared by **both harnesses** (Claude sees them as `mcp__pupa_mcp__*`). Playwright also needs Node.js + `npx --yes playwright install chromium` (`make install-playwright`). |
| **Subagent delegation** | _(env only)_ | `PUPA_SUBAGENTS_DISABLED=1` | **On by default.** Adds a `task` tool for delegating to specialist subagents. |
| **Agent skills** | `skills_disabled: true` to turn off | `PUPA_SKILLS_DISABLED=1` | **On by default.** Progressive disclosure over `SKILL.md` workflows under `~/.pupa-backend/skills/` (none shipped; dir starts empty); names + descriptions in the prompt, bodies loaded on demand via `skill_view`. Pinned off in the cloud image. |

Env-only tuning knobs: `LG_RECURSION_LIMIT` (default 100) ·
`LG_CLEAR_TOOL_USES_TRIGGER` (default 40000). Persistence lives under
`config.yml`'s `persistence:` block, or override with a single
`DATABASE_URL` (see [docs/architecture.md](docs/architecture.md)).

## Architecture

The client speaks AG-UI to the backend over one `POST /` SSE stream. The backend
runs the **agent harness** chosen for that connection; the harness talks to a
model and, when it wants one of your app's tools, round-trips back to the device
to run it. Auth, persistence, MCP, and the screen-share broker sit around the
harness and are the same whichever one is active.

```
                     AG-UI  ·  POST /  ·  SSE
   iOS / macOS  ───────────────────────────────▶  pupa-backend  (FastAPI :8004)
        ▲                                                 │
        │  tool call ─▶ device runs it ─▶ result          │  active harness ─▶ a model
        └─────────  AG-UI interrupt / resume  ────────────┘

   Agent harness  (chosen per connection):
     • Claude Code loop         →  your Claude subscription
     • Deepagents (LangGraph)   →  Bedrock / Anthropic / OpenAI-compatible
     • … more to come

   Harness-independent:  pair-once auth · your forwarded tools · MCP servers ·
   persistence (SQLite / Postgres / in-memory) · /screenshare/ws
```

Full reference — per-harness detail, per-request model swap, auth flow,
screenshare broker, tool gating — in
[docs/architecture.md](docs/architecture.md).

## Auth model

The backend **requires auth by default**. The only client credential is a
paired-device token in the iOS Keychain.

- **Bootstrap.** Set `PUPA_API_KEY` server-side, run `make pair` → 8-char
  code, paste / scan in iOS Settings → Backend → Edit.
- After the first pair, the operator can `unset PUPA_API_KEY` — paired
  devices keep working; fresh devices get 401 until the next pair.
- The key never reaches a client.
- **Disable auth entirely**: `PUPA_AUTH_DISABLED=1`. Dev loops only
  (`make backend` + the macOS client on the same laptop); never expose
  this to a reachable backend.

Token store backends:

- **Local / SQLite**: JSON file at `backend/pupa-auth.json` (override via
  `PUPA_AUTH_DB_PATH`).
- **Postgres**: `PostgresDeviceStore` auto-selects when the checkpointer
  is Postgres-backed; tokens survive ephemeral filesystems (Railway).

## Connecting the app

The phone reaches the backend over whatever network path you have — pick one.
`pupa-backend pair` (or `make pair`) auto-derives the right URL for that path and
bakes it into the QR code, so the app connects straight away.

- **Local network** — phone and backend on the same Wi-Fi (or the same Mac).
  Nothing to set up; this is the default and perfect for a laptop next to you.
- **[Tailscale](https://tailscale.com)** — put the backend on your private
  Tailscale network and reach it from anywhere as if it were local, no ports
  opened. Set `connectivity: tailscale` (`PUPA_CONNECTIVITY=tailscale`); pairing
  uses the Tailscale address.
- **[Cloudflare](https://www.cloudflare.com/products/tunnel/) tunnel** — expose
  the backend over HTTPS without opening a port:
  Set `connectivity: cloudflared`; `pupa-backend run` starts `cloudflared` for
  you.
  - *Quick tunnel* (no signup): throwaway `trycloudflare.com` URL, changes on
    every restart — re-pair each time.
  - *Named tunnel* (stable URL on your own domain): configure once with
    `pupa-backend setup` (cloudflared login + your domain); pair once.

Auth still applies whichever path you use — the app carries its paired-device
token (see [Auth model](#auth-model)). For a remote deployment you can skip
auto-derivation and pass the URL explicitly:
`pupa-backend pair --public-url https://<your-host>`.

## Screen-share sidecar

[`screenshare-sidecar/`](screenshare-sidecar/) is an optional macOS-only
Swift CLI that captures a window via ScreenCaptureKit and publishes the
video as a WebRTC track to the broker mounted at `/screenshare/ws` on
this backend. Viewer is either the iOS app or the bundled browser viewer
at [`screenshare-sidecar/viewer/`](screenshare-sidecar/viewer/).

```bash
pupa-backend screenshare
```

The sidecar is a separate Swift package — `swift build --package-path
screenshare-sidecar` from a fresh clone is enough to compile it. macOS 14
+ Xcode toolchain required.

## Deploy

[`docs/deploy.md`](docs/deploy.md) — Railway + Postgres + Langfuse with
the multi-tenant safety posture baked in (`shell_tool_enabled: false`,
`screenshare: false`, no `mcp_servers`). The image is built from
[`Dockerfile`](Dockerfile) at the repo root; [`railway.json`](railway.json)
points at it.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the branching workflow (`dev`
integration, fast-forward release to `main`) and the hard rules for AI
assistants. [CLAUDE.md](CLAUDE.md) is the in-repo agent guide.

## Built on

Pupa backend stands on open-source shoulders. The keystone is the
**[AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui)** — the single
contract the iOS / macOS client and this backend share over `POST /` (SSE).
Everything else plugs into that contract:

| Project | Role here |
|---|---|
| **[AG-UI](https://github.com/ag-ui-protocol/ag-ui)** | **The key dependency** — the agent ↔ UI event protocol; the only language the client and backend speak. |
| **[Claude Code](https://github.com/anthropics/claude-code)** + **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)** | The Claude Code loop harness — runs the `claude` CLI in-process for subscription-billed agent turns. |
| **[CopilotKit](https://github.com/CopilotKit/CopilotKit)** | AG-UI runtime + middleware that bridges the deepagents graph to the protocol (`ag-ui-langgraph`, `copilotkit`). |
| **[deepagents](https://github.com/langchain-ai/deepagents)** | The deepagents (LangGraph) harness — agent, subagents, skills, and filesystem middleware. |
| **[LangGraph](https://github.com/langchain-ai/langgraph)** / **[LangChain](https://github.com/langchain-ai/langchain)** | The agent graph, middleware, checkpointer, and store under deepagents. |
| **[Langfuse](https://github.com/langfuse/langfuse)** | LLM tracing / observability — on whenever Langfuse credentials are set, off with `PUPA_LANGFUSE_DISABLED=1`. |
| **[FastAPI](https://github.com/fastapi/fastapi)** + **[Uvicorn](https://github.com/encode/uvicorn)** | The ASGI app and server. |
| **[Tavily](https://github.com/tavily-ai/tavily-python)** | Optional web-search backend tool. |
