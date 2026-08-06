# Changelog

All notable changes to the Pupa backend repo are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — patch-only
bumps (`0.0.X` → `0.0.X+1`).

## [0.0.44] — 2026-08-06

Baseline entry. This repository was reset at this commit; entries above it
describe changes, this one describes the starting point.

### The backend at this commit

- **AG-UI over a single SSE stream.** `POST /` is the whole protocol surface a
  client needs. Two transport middlewares wrap every stream: `SSEReplay`
  (per-thread sequenced replay, so a dropped socket can re-attach mid-turn) and
  `SSEKeepAlive` (idle `: keep-alive` comments, so a silent turn doesn't trip
  the client's request timeout).
- **Two agent harnesses, mounted side by side.** Every enabled harness serves
  `POST /harnesses/{id}`; the default one is also aliased at `POST /`.
  `GET /harnesses` is the discovery document — each harness's model menu,
  toggleable tools, and permission-control schema, so the client renders the
  right UI without an app update.
  - `deepagents` — a LangGraph `create_agent` graph with per-request model
    selection, checkpointed threads, and frontend tools dispatched via
    `langgraph.interrupt()`.
  - `claude_code` — the Claude Code SDK loop, subscription-billed, with
    frontend tools bridged as an in-process MCP server and native host tools
    behind a `PreToolUse` permission gate.
- **Frontend tools belong to the client.** The backend forwards their
  JSON-Schema definitions to the model and the client executes the calls. The
  only backend tools are `tavily_search`, an env-gated `shell`, `write_todos`,
  and whatever MCP servers are configured.
- **Pair-once auth, required by default.** Bootstrap with a server-side
  `PUPA_API_KEY`, run `make pair` for an 8-char code, and the device holds a
  token in the iOS Keychain from then on. The key never reaches a client.
- **One `DATABASE_URL` drives persistence** — its scheme selects the backend
  for both the LangGraph checkpointer and store. Unset falls back to SQLite
  under `~/.pupa-backend/`; cloud deploys pin `PUPA_REQUIRE_DB_SCHEME` to
  forbid that fallback.
- **Optional extras:** Langfuse tracing (on whenever credentials are present),
  a macOS screen-share sidecar publishing WebRTC into `/screenshare/ws`, and
  config-driven MCP servers shared by both harnesses.

### Versions at this baseline

| Package | Version |
|---|---|
| Python backend (`backend/pyproject.toml`) | `0.0.81` |
| Screen-share sidecar | `0.0.6` |

Releases before this entry are on PyPI as `pupa-backend` through `0.0.81`;
their history is not carried over here.
