# Changelog

All notable changes to the Pupa backend repo are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — patch-only
bumps (`0.0.X` → `0.0.X+1`).

## [0.0.90] — 2026-08-21

### Security

- **Devices can no longer pair other devices.** Minting a pairing code
  (`/auth/pair/begin`) now requires the operator key. Before, any paired-device
  token was accepted, which meant a leaked token could quietly issue itself a
  replacement — and revoking the device it came from wouldn't end the access.
  **Keep `PUPA_API_KEY` set:** it is now the only credential that can pair a
  device, so unsetting it after the first pair means you can't add a second.
- **Pairing requests asked for their own permissions, and got them.** The
  scope list in a `/auth/pair/begin` body was used as sent. It's now checked
  against the known set, and the route it arrives on is operator-only.
- **Running the agent now requires the `agent` scope.** `POST /` and
  `POST /harnesses/{id}` previously accepted any valid token regardless of the
  scopes it was issued with — which is the most powerful thing a token can do,
  since it reaches every backend tool.
- **Failed pairing attempts are rate limited.** `/auth/pair` is the one route
  a stranger can call, and nothing stopped them calling it at full speed. Now
  5 wrong codes per client per minute (and 10 wrong operator keys on
  `/auth/pair/begin`), with `PUPA_RATE_LIMIT_DISABLED=1` for local loops.
  **Only failures count** — pairing a device successfully, or minting codes
  with the operator key, is never throttled, so the limit falls on guessing
  rather than on using the thing. There's no shared cap either: one that
  everybody drew from could be drained by a stranger to lock out people
  holding real credentials. Bucketing reads `X-Forwarded-For` only where a
  proxy is trusted — every tunnel mode terminates in front of a loopback
  listener, so the peer address is `127.0.0.1` for everyone there, while on a
  direct bind that header is written by the caller and believing it would hand
  out an unlimited supply of buckets.
- **New `PUPA_TRUSTED_PROXY`.** Says whether `X-Forwarded-*` can be believed
  on this deployment. Inferred for the Tailscale and Cloudflare modes and
  pinned on in the cloud image, so working setups stay working; off everywhere
  else, because that's the answer that can't be turned against you. uvicorn's
  own forwarded-header handling is switched off so this stays the only answer
  — it believed those headers from any loopback peer, which on a tunnel
  deployment is every caller. **If you run your own reverse proxy in front of
  the backend, set `transport.trusted_proxy: true`**; the modes Pupa starts
  itself are inferred and need nothing. `connectivity: cloudflared` bound
  `0.0.0.0` while trusting `X-Forwarded-*`, so anyone on the LAN could forge a
  hop; it now binds loopback, and Tailscale's raw-TCP fallback mode — which
  passes the client's request through byte-for-byte and writes no forwarded
  headers — is no longer trusted to have written them.
- **Forwarded headers are read across every field line.** A proxy that
  *appends* `X-Forwarded-For: <client>` as a second header line rather than
  extending the caller's left the whole value caller-controlled, because only
  the first line was parsed — so "the rightmost hop is the proxy's" was not
  what the code did.
- **Concurrent pairing guesses are capped.** The limiter checked a caller's
  budget before running the request and charged it after, so requests arriving
  together all passed the check against a bucket none of them had been written
  to yet. The charge now goes on first and comes back off when the response
  says the caller was legitimate.
- **A pairing request asking for no scopes gets none.** `"scopes": []` in a
  `/auth/pair/begin` body was read as "unspecified" and minted a device with
  the full default set, `agent` included.
- **New `PUPA_REQUIRE_HTTPS` refuses plaintext.** Off by default so LAN and
  offline installs keep working; **set it on anything reachable from the
  internet**, where it's the difference between handing the device token to the
  client and handing it to the network. Pinned on in the cloud image. Covers
  the screen-share socket too.
- **The shell tool no longer inherits your API keys.** `SHELL_PASS_ENV=1`
  handed the backend's whole environment to a subprocess the *model* drives,
  filtered only by a list the operator had to write themselves. Secret-shaped
  names (`*_API_KEY`, `*_TOKEN`, `AWS_*`, …) are now dropped by default;
  `SHELL_ENV_ALLOW` names exceptions.
- **`transport.trusted_proxy: false` now takes effect.** A `false` in
  `config.yml` was dropped rather than written out, and an unset
  `PUPA_TRUSTED_PROXY` means "infer from `connectivity`" — so on a Tailscale or
  Cloudflare deployment, the one place you'd turn proxy trust off didn't.
- **Responses carry security headers** — `nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, and HSTS on TLS connections — including the
  tunnel modes, where the terminator holds the cert and the backend speaks
  plain HTTP over loopback. Still no CORS: there's no browser origin to allow.
- **33 known vulnerabilities in dependencies, fixed.** The lockfile carried
  advisories against `starlette`, `aiohttp`, `cryptography`, `mcp`,
  `python-multipart` and others. Upgraded to zero. CI now runs `pip-audit` and
  a secret scan weekly and on every PR, so the next one surfaces in review.
  Neither job is a required check, but neither is `continue-on-error` either —
  GitHub reports those as passing, which would have made a new advisory look
  identical to a clean run.

### Changed

- **The HTTPS guard now sits outside the rate limiter.** A plaintext request is
  refused before the limiter sees it, so it can't spend a real device's pairing
  budget — the limiter no longer needs to know anything about transport.

### Fixed

- **A refused screen-share socket reports why.** The 4403 / 4401 / 4400 close
  codes were sent on a socket that had never been accepted, so a real server
  answered the upgrade with a bare HTTP 403 and the client couldn't tell an
  HTTPS problem from an auth one.
- **A frontend tool call no longer stops the chat when it lands second.**
  `ag-ui-langgraph` 0.0.43 fixes the emit path that only looked for parked
  interrupts on the first task — when one parked elsewhere, the turn ended
  looking exactly like a clean finish and the chat sat silent until the user
  typed again.

## [0.0.89] — 2026-08-20

### Added

- **Token and cache usage now show up in the server log.** The Claude loop
  prints a yellow line for every model call — input, output, cache reads, cache
  writes, and how much of the prompt came from cache — plus a totals line at the
  end of each turn with the cost, the number of model turns, and the time spent
  waiting on the API.
- **The log now says why a turn re-paid for its cache.** Each turn also prints
  what changed since the thread's last one — the model, the tool set, or a named
  piece of the context the app sends every turn (live canvas state, memories,
  skills). A turn that changed nothing says so, and should be cheap.
- **Opt-in prompt dump for chasing a cache miss.** Point
  `PUPA_CLAUDE_PROMPT_DUMP` at a scratch folder and each turn writes down exactly
  what was sent to the model, alongside a diff against the previous turn — so
  "nothing changed but it re-charged me" is answerable from the file rather than
  guessed at. Off unless you set it: the files hold your app's own content.

### Fixed

- **Turns stopped re-paying for context that never changed.** The app describes
  itself to the model every turn — canvas, memories, skills, subagents — and the
  same description was arriving with its fields written in a different order each
  time. Nothing had changed, but the model had to be re-told everything behind
  it, including the whole conversation so far. That was roughly half the cost of
  a short turn, and it grew as the chat got longer. The backend now reads those
  payloads in a fixed order, so an unchanged app costs nothing to re-describe —
  including for app versions already installed.

## [0.0.88] — 2026-08-16

### Fixed

- **A retried request no longer splits a reply in two.** When a client re-sends
  a request whose response was lost, the second one could land while the first
  was still streaming — and the two then took turns pulling from the same queue,
  so each got a fragment of the reply and the thread's saved copy came out in the
  wrong order. The newer request now takes the stream over cleanly, and anything
  the older one was holding is handed across rather than dropped.

## [0.0.87] — 2026-08-15

### Fixed

- **Re-attaching now counts as proof the app is alive.** A re-attach POST is
  served by the replay middleware without reaching an agent loop, so a turn
  parked on that thread never heard about it: a client that had reported itself
  backgrounded (which suspends the liveness grace) stayed marked backgrounded on
  a stale last-ping clock at the very moment it reconnected, and a parked
  frontend tool could still be failed by the grace. The middleware now fires
  `notify_reattach`, and the Claude loop re-arms the session's liveness clock
  from it. Observers are best-effort, so a failing hook can't cost the client its
  replay tail.

## [0.0.86] — 2026-08-15

### Fixed

- **Sending a new message on a thread parked mid-tool no longer breaks the
  turn.** Reopening the app and typing something new tore the parked Claude Code
  session down by cancelling its pump and closing the SDK transport outright.
  That rejected the CLI child's in-flight PreToolUse / permission roundtrip
  (`Error in hook callback hook_0: … Stream closed`) and left the SDK session
  interrupted, so the next turn's `resume` was answered with the CLI's
  `Continue from where you left off.` no-op — the user's actual prompt sat queued
  behind it and the turn produced nothing. A new-turn POST now retires the parked
  session first: release its waiting tool handlers, `interrupt()` the child, wait
  a bounded window for the turn to wind down, then dispose. Tunable with
  `PUPA_CLAUDE_RETIRE_DRAIN` (default 2s); every step is best-effort, so a wedged
  child still frees the thread.

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
