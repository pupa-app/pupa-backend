# Changelog

All notable changes to the Pupa backend repo are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — patch-only
bumps (`0.0.X` → `0.0.X+1`).

## [0.0.85] — 2026-08-15

### Fixed

- **Silent empty turns in the Claude Code loop now explain themselves.** The pump
  suppressed the text of every whole `AssistantMessage`, on the assumption it had
  already streamed as deltas. But the CLI fabricates some assistant messages
  locally and those never stream — rate-limit notices ("You've hit your session
  limit · resets …"), API errors, and the `No response requested.` reply it gives
  a prompt queued behind a resumed session's `Continue from where you left off.`
  Their text was deleted, so the run emitted `RUN_STARTED` + `RUN_FINISHED` and
  nothing else and the app rendered it as a dropped connection — most visibly
  after waking the app on a thread whose previous turn was torn down mid-tool.
  Suppression is now decided per message from the ids that actually streamed, so
  a non-streamed message keeps its text.

## [0.0.84] — 2026-08-12

### Fixed

- **Tailscale deploys are reachable again on macOS.** macOS gates connections to
  `0.0.0.0`-bound sockets behind the Local Network privacy grant; without it
  nothing reached the backend — not the phone, not `pupa-backend pair` on the
  same machine (SYNs stalled in `SYN_RCVD`). With `connectivity: tailscale` the
  backend now binds `127.0.0.1` and publishes itself to the tailnet via
  `tailscale serve`, torn down on shutdown. When the tailnet has HTTPS enabled,
  serve terminates TLS on :443 with a real auto-renewing Let's Encrypt cert for
  the MagicDNS name and the backend speaks plain HTTP on loopback — **the
  device pairs against a normally-trusted URL with no cert fingerprint at all**.
  Without tailnet HTTPS it falls back to raw-TCP passthrough (self-signed cert
  end-to-end, fingerprint pinned) and logs how to enable HTTPS.
  `PUPA_TAILSCALE_SERVE=0` opts out, `=tcp`/`=https` forces a mode, and
  `PUPA_HOST` overrides the bind address.
- **Self-signed certs iOS will actually accept.** `pupa-backend setup` minted
  10-year certs; Apple refuses any TLS server cert valid for more than 398 days
  and any cert without the `serverAuth` EKU, rejecting the connection *before*
  the client's fingerprint pinning runs — pairing failed with "the backend
  refused a secure connection". New certs are 397 days with the EKU set, and
  startup warns when the configured cert is over-long, expired, or within 30
  days of expiry. **Existing tailscale/localhost setups must re-run
  `pupa-backend setup` and re-pair their devices** (the cert, and so the pinned
  fingerprint, changes).

## [0.0.83] — 2026-08-11

### Added

- **Per-turn extended-thinking level for the Claude Code loop.** The client
  picks a level via `forwardedProps.llm.thinking` (`auto`/`off`/`low`/`medium`/
  `high`); the loop maps it to `ClaudeAgentOptions.thinking` (`adaptive` /
  `disabled` / `enabled` with an ascending token budget). No key → option unset
  (CLI default), so existing threads are unchanged. `GET /harnesses` now reports
  a `thinking` menu per harness (`[]` for harnesses without the capability).

## [0.0.45] — 2026-08-07

### Added

- **`make pair` now takes the flags the underlying CLI always had.** `URL=` for
  a remote deploy, `KEY=` for that backend's `PUPA_API_KEY`, and `CODE_TTL=` /
  `DEVICE_TTL=` for the bootstrap-code and device-token lifetimes. Pairing a
  phone against Railway previously meant a raw `curl` — which returns JSON and
  therefore no QR code, so the URL and code had to be typed by hand.

### Fixed

- The deploy runbook described the baked cloud config as pinning
  `langfuse.enabled: true`. No such key exists — tracing is opt-out, active as
  soon as `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are present, with
  `langfuse.disabled` as the only switch.

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
