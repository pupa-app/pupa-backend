# Architecture

Snapshot of how the backend is wired right now. Read this before changing
agent behaviour, persistence rules, auth, or the screen-share broker.

## Packaging & CLI

The backend is the `pupa-backend` distribution on PyPI (name in
[`backend/pyproject.toml`](../backend/pyproject.toml)); all Python code lives
under the `pupa_backend/` package. Installing it (`pip`/`pipx install
pupa-backend`) puts a `pupa-backend` console script on PATH via
`[project.scripts] pupa-backend = "pupa_backend.cli:main"`
([`backend/pupa_backend/cli.py`](../backend/pupa_backend/cli.py)) — the
in-process dispatcher for `run` / `stop` / `status` / `pair` / `setup` / `mcp`
/ `service-*` / `logs` / `screenshare`. `run`/`stop`/`status` track a
`~/.pupa-backend/pupa.pid` pidfile.

The server process is launched as `python -m pupa_backend.app` (Makefile,
Dockerfile, Railway `startCommand`, and the generated launchd/systemd unit in
[`service.py`](../backend/pupa_backend/scripts/service.py)); `pupa-backend run`
calls the same `pupa_backend.app:main`.

The two differ in **environment**. `run` is a child of your shell and inherits
its exports; the service inherits nothing and reads `~/.pupa-backend/config.yml`
itself (`app.py` calls `load_pupa_config()` at import). The generated unit
therefore carries **only `PATH`** — launchd/systemd start with a minimal one and
the backend can't reconstruct the operator's, so that single var has to be
passed in. Nothing else is snapshotted: the unit would only duplicate what the
process already reads, the plist is world-readable where config.yml is `0600`,
and a snapshot goes stale the moment config.yml is edited.

The consequence is that a credential exported only in a shell profile works
interactively and is invisible to the service. `service-install` therefore
refuses when it finds one, naming each var and the config.yml key it belongs
under (`_assert_no_shell_only_secrets`; bypass with
`PUPA_SERVICE_ALLOW_SHELL_ONLY=1`), rather than writing a unit that crash-loops
in a log file. The var list is derived from `pupa_config.known_env_vars()` —
the same maps the loader writes through, so it can't drift — plus whatever
`service.check_env` adds. The version an app sees at
`GET /auth/config` is `importlib.metadata.version("pupa-backend")` — the exact
installed release, so a client can pin a compatible backend with
`pipx install pupa-backend==0.0.X`. Publishing to PyPI is automatic: a push to
`main` runs `.github/workflows/publish.yml` (Trusted Publishing / OIDC), which
publishes only when `backend/pyproject.toml`'s version is not already on PyPI
and then pushes a `v<version>` provenance tag — no manual tagging. The
screen-share sidecar is **not** in the wheel — it stays a source-built Swift
package.

## App shape

[`backend/pupa_backend/app.py`](../backend/pupa_backend/app.py) builds the FastAPI app with a
single lifespan that:

1. Loads YAML config from `~/.pupa-backend/config.yml` (and the legacy
   `.env` fallback) via
   [`backend/pupa_backend/pupa_config.py`](../backend/pupa_backend/pupa_config.py). Shell env
   always wins.
2. Spins up a
   [`db.open_persistence`](../backend/pupa_backend/harnesses/langgraph/db/connection.py)
   from `DATABASE_URL` (or local SQLite when unset) and yields the
   checkpointer + store to the graph — **only when the deepagents harness is
   enabled**. A Claude-only deploy opens no database and mounts no `/db`;
   that loop keeps sessions in-process and the SDK owns its history.
3. Pre-builds the env-default agent graph and registers it under the
   `(None, None)` cache key via
   [`agent.register_graph_deps`](../backend/pupa_backend/harnesses/langgraph/agent.py).

Routers mounted on the app:

| Prefix              | Module                                                          | Purpose                                                            |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------ |
| `POST /`               | default [harness](#agent-harnesses-multiple-mounted-together)   | AG-UI SSE stream — the only protocol the client speaks. Alias for the default harness (un-migrated clients). |
| `POST /harnesses/{id}` | the harness with that id                                        | AG-UI SSE stream for a specific harness (`deepagents`, `claude_code`). |
| `GET /harnesses`       | [`backend/pupa_backend/harnesses/routes.py`](../backend/pupa_backend/harnesses/routes.py) | Discovery: each enabled harness's models, tools, and permission-control schema. Replaces `/models` + `/backend-tools`. |
| `/auth/*`           | [`backend/pupa_backend/auth/routes.py`](../backend/pupa_backend/auth/routes.py)           | Pair-once flow: `/auth/pair/begin`, `/auth/pair/complete`, `/auth/devices`, `/auth/config`. |
| `/db/*`             | [`backend/pupa_backend/harnesses/langgraph/db/routes.py`](../backend/pupa_backend/harnesses/langgraph/db/routes.py) | Transcript loader, per-thread usage, thread deletion. Mounted only when the deepagents harness is enabled. |
| `/screenshare/ws`   | [`backend/pupa_backend/screenshare/routes.py`](../backend/pupa_backend/screenshare/routes.py) | WebRTC signalling broker (opt-in, see env).                        |

Healthcheck path for Railway: `GET /auth/config`.

**SSE keep-alive.**
[`SSEKeepAliveMiddleware`](../backend/pupa_backend/sse_keepalive.py) wraps every
`text/event-stream` response and emits an SSE comment (`: keep-alive`)
whenever the upstream generator is silent for 15s
(`PUPA_SSE_KEEPALIVE_INTERVAL`, `<= 0` disables). It's a transport
concern — the iOS `URLSession` idle timeout (`timeoutIntervalForRequest`,
~60s) only resets on received bytes, so a long silent turn would
otherwise surface as `NSURLErrorDomain -1001`. Living at the app
boundary, it covers every harness run stream (`POST /` and
`POST /harnesses/{id}`) and any future SSE route with no per-route code.

**Resumable SSE (replay + re-attach).**
[`SSEReplayMiddleware`](../backend/pupa_backend/sse_replay.py) is the other
transport-level concern, and the reason a backgrounded app doesn't lose a turn.
It *detaches* the handler's body iterator from the HTTP response: a background
pump drains the agent's generator to completion, stamping each SSE frame with a
monotonic per-thread sequence number (the SSE `id:` field) into an in-memory
`ReplayLog` ring, while the response body is merely a tail reader over that log.
A client disconnect closes the tail only — **the run keeps going**.

That last sentence is why the middleware is plain ASGI rather than a
`BaseHTTPMiddleware`: the Starlette base runs the inner app inside the
*request's* task group, so a disconnect cancels the handler and closes the
stream the pump is draining. Detaching a run cannot be done from inside a scope
the client can collapse. For the same reason the handler is given a `receive`
that never reports `http.disconnect` — `StreamingResponse` races the body
against `listen_for_disconnect` and cancels it the moment one arrives. The
client's own connection is watched by the tail response, where hanging up is
harmless.

A client that lost its socket re-attaches by POSTing
`forwardedProps.command.reattach.after_seq`; the middleware short-circuits that
request before either agent loop sees it and replays every frame with a higher
seq, then live-tails. `after_seq = -1` replays the whole turn, so an app killed
mid-run catches up from the buffer on relaunch. An unknown thread answers
`204`. Response headers (`X-Pupa-Replay-Live` / `-Next-Seq` / `-Oldest-Seq`)
tell the client whether a pump is still attached and what the log still holds.

Knobs: `PUPA_SSE_REPLAY_TTL` (default 6h; `<= 0` disables),
`PUPA_SSE_REPLAY_MAX_EVENTS` (ring cap per thread, default 4096).

**Middleware order is load-bearing**, and reads bottom-up in
[`app.py`](../backend/pupa_backend/app.py) because `add_middleware` prepends.
Outermost to innermost: security-headers → require-https → rate-limit →
api-key → run-scope → keep-alive → replay → handler.

Why that order, at each step:
- **security-headers** outermost so the guards' own 403/429 responses carry
  the headers too — those never reach the inner stack.
- **require-https** outside the limiter: a plaintext hop is a
  misconfiguration, not a guess at a credential, so refusing it there is what
  stops it spending a real device's pairing budget. The limiter then needs to
  know nothing about transport.
- **rate-limit** before auth: it protects `/auth/pair`, which is *pre*-auth, so
  it has to apply regardless of the auth outcome.
- **run-scope** inside api-key: it reads the `request.state.auth` identity that
  api-key resolves.
- **replay** innermost so heartbeat comments stay out of the replay log, while
  an idle re-attached stream still receives them.

`tests/test_middleware_wiring.py` pins this against the real app — the other
auth tests build their own and would keep passing if a guard were unmounted.

Two accepted limits: the log is in-memory, so a backend restart drops it; and
recovery depends on the client persisting its last seq — a client that loses
that value across a process kill can only fall back to a normal send.

## Agent graph

[`backend/pupa_backend/harnesses/langgraph/agent.py`](../backend/pupa_backend/harnesses/langgraph/agent.py) builds a LangGraph
`create_agent` with `CopilotKitMiddleware` +
[`ToolGatingMiddleware`](../backend/pupa_backend/harnesses/langgraph/tool_gating.py). Tools available to
the model:

- **Frontend tools** — the client ships JSON-Schema definitions on every
  request; the backend forwards them to the model as proper tool
  definitions. The client executes the calls. Don't enumerate these
  tools in the system prompt — names, schemas, and descriptions are
  already forwarded.

  Each frontend call pauses the graph via `langgraph.interrupt()`
  ([`frontend_interrupt.py`](../backend/pupa_backend/harnesses/langgraph/frontend_interrupt.py)); the client
  runs it and resumes with `Command(resume=…)`. **Known upstream bug:**
  `ag-ui-langgraph` (pinned `0.0.42`) emits `on_interrupt` from
  `state.tasks[0]` only, so a multi-task turn (batched `render*`, or `render*`
  + a backend tool) whose interrupt parks on a later task drops it in-run —
  the run looks like a clean finish and the chat stalls until the next POST,
  whose recovery path collects all tasks. Fixed on upstream `main`, unreleased.
  The iOS client self-heals by re-POSTing on this signal; a canary
  (`tests/test_ag_ui_langgraph_emit_interrupt_bug.py`) fails when the fix ships.
- **Backend tools** — `tavily_search` (when `TAVILY_API_KEY` is
  set; see [`backend/pupa_backend/harnesses/langgraph/backend_tools.py`](../backend/pupa_backend/harnesses/langgraph/backend_tools.py))
  and the optional local `shell` tool (gated on `SHELL_TOOL_ENABLED=1`)
  guarded by the approval gate below. Browser automation (Playwright) is
  no longer special-cased — it is an ordinary MCP server (below).
- **`claude_code`** — delegates a self-contained coding/research job to a
  full Claude Code agent by shelling out to the `claude` CLI in headless
  mode (`claude -p … --output-format json`) and returning the single
  synthesized result; the spawned agent runs in its own process with its
  own filesystem-aware tool loop. One-shot transport (the CLI persists
  sessions, so `resume_session_id` → `--resume` gives multi-turn
  continuity without a long-lived process). **Read-only by default**
  (`mode="plan"` → `--permission-mode plan` plus `--allowedTools Read Grep
  Glob`); `mode="edit"` opts into writes (`acceptEdits`). The subprocess
  gets a **minimal env** (PATH/HOME + the Anthropic/Bedrock credential vars
  `claude` itself needs), not the backend's full environment — set
  `CLAUDE_CODE_PASS_ENV=1` to forward everything (minus
  `_shell_env_exclude()`). On by default for local installs; opt out with
  `PUPA_CLAUDE_CODE_DISABLED=1` (config `claude_code_disabled: true`), and
  **pinned off in the multi-tenant cloud**. Other knobs: `CLAUDE_CODE_BIN`,
  `CLAUDE_CODE_MODEL` (config `claude_code_model`), `CLAUDE_CODE_WORKSPACE`
  (config `claude_code_workspace`), `CLAUDE_CODE_TIMEOUT`,
  `CLAUDE_CODE_MAX_TURNS`. See
  [`backend/pupa_backend/harnesses/langgraph/claude_code_tool.py`](../backend/pupa_backend/harnesses/langgraph/claude_code_tool.py).
- **MCP servers (config-driven)** — any number of MCP servers declared
  under `mcp_servers:` in `~/.pupa-backend/config.yml`
  ([`backend/pupa_backend/mcp_servers.py`](../backend/pupa_backend/mcp_servers.py)). The block
  (shape mirrors Claude Code's `.mcp.json` and
  `langchain-mcp-adapters`' `MultiServerMCPClient`) is serialised to the
  `PUPA_MCP_SERVERS` env var by
  [`backend/pupa_backend/pupa_config.py`](../backend/pupa_backend/pupa_config.py); a lifespan-managed
  lifecycle loads each enabled server's tools. **Every server's tools start
  hidden behind a single `get_tools(server=…)` gate tool** whose description
  lists the available servers (with each server's optional `description:`); an
  `McpGateMiddleware` reveals a server's tools only after the agent activates
  it for that thread, keeping the model's tool list lean no matter how many
  servers are registered. A failing server is logged and skipped. Shipped
  presets: Atlassian Confluence via the community `mcp-atlassian` stdio server
  (`uvx mcp-atlassian`, static API token), and Playwright browser automation
  via `@playwright/mcp` (`npx`; browser binaries from `make install-playwright`).
  The MCP client deps (`langchain-mcp-adapters`, `fastmcp>=3.4.2`) ship by
  default in core dependencies — no extra to install. Each
  configured server surfaces as an `mcp_<name>` entry in the deepagents harness's
  `tools` list on `GET /harnesses` for iOS Settings discovery. Manage the block with the `pupa-backend mcp
  add|list|remove` CLI ([`backend/pupa_backend/scripts/mcp.py`](../backend/pupa_backend/scripts/mcp.py),
  logic in [`backend/pupa_backend/mcp_config_admin.py`](../backend/pupa_backend/mcp_config_admin.py),
  presets: `--playwright`, `--confluence-url/--confluence-user`) instead of
  editing YAML by hand.
- **Agent skills** — the skill lifecycle
  ([`backend/pupa_backend/harnesses/langgraph/skills.py`](../backend/pupa_backend/harnesses/langgraph/skills.py), on by default; off with
  `PUPA_SKILLS_DISABLED=1`). See [Skills](#skills) below.

### Per-request model swap

The client may include `forwardedProps["llm"] = {"provider", "model"}`
in `RunAgentInput`. The
[`CustomLangGraphAGUIAgent`](../backend/pupa_backend/harnesses/langgraph/harness.py) validates it
through `LLMParams` (Pydantic, `extra="forbid"`) and calls
`_resolve_per_request_graph` to look up a cached graph for the
`(provider, model)` pair via `MODEL_REGISTRY`. `self.graph` is swapped
for the request lifetime (restored in `finally`). Unknown pair or
missing creds → `RunErrorEvent(code="llm_unavailable")` so the iOS chat
surfaces a toast.

Caching: `get_model()` and `get_graph()` are both keyed by
`(provider, model_id)` so the same pair reuses one model instance and
one compiled graph across requests.

Providers in `MODEL_REGISTRY`:

- **Bedrock** (`langchain_aws.ChatBedrockConverse`, EU inference
  profiles) — Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5.
- **Anthropic** (`langchain_anthropic.ChatAnthropic`, direct API) —
  Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5.
- **OpenRouter** (`openrouter`) — a curated menu of popular open /
  frontier slugs (GLM, Qwen, MiniMax, Kimi, DeepSeek) served through
  OpenRouter's OpenAI-compatible endpoint. Auth is just
  `OPENROUTER_API_KEY` in the backend env. Usable as a per-request
  picker choice **and** as the default graph: set
  `LLM_PROVIDER=openrouter` + `LLM_MODEL=<slug>` (the config wizard's
  `openrouter` provider type writes both, leaving the key in the shell).
  The model slug for the default path is *not* registry-bound — any
  valid slug works. Verify slugs at <https://openrouter.ai/models>.

Separately, the env-pinned **`openai_compatible`** provider
(`LLM_PROVIDER=openai_compatible` + `LLM_BASE_URL` / `LLM_API_KEY` /
`LLM_MODEL`) routes the default graph to any single OpenAI-compatible
endpoint; it has no registry entries because each deployment points at a
different proxy.

## Agent harnesses (multiple, mounted together)

An **agent harness** is a self-contained agent loop that owns an AG-UI SSE
handler. Two ship today — the **deepagents** graph and the **Claude Code** loop
([`backend/pupa_backend/harnesses/claude/`](../backend/pupa_backend/harnesses/claude/)) — and *every enabled harness
is mounted at once* at `POST /harnesses/{id}` (the default one is also aliased at
`POST /` for un-migrated clients). The iOS app picks the harness per backend
connection; the server no longer runs one loop per process.

The registry lives in [`harnesses/__init__.py`](../backend/pupa_backend/harnesses/__init__.py) (`AgentHarness`
protocol + `DeepAgentsHarness` / `ClaudeCodeHarness` adapters), built from the
`PUPA_HARNESSES` env JSON that `pupa_config` emits from the config.yml
`harnesses:` block. `app.py`'s lifespan loops over `registry.enabled()` and calls
`harness.register(app, path, deps)`. Adding a harness = adding an adapter (a
public plugin entry point is deferred until the package is published).

**Coexistence of deepagents + Claude Code.** The Claude loop drives the `claude`
CLI, which inherits `os.environ`; its billing guard is subscription-only and
refuses to start if `ANTHROPIC_API_KEY` / `AWS_*` are present (the subprocess
would inherit them and bill per-token API credits). But the deepagents harness
*needs* those keys. So when the Claude harness is enabled, a **credential stash**
([`credentials.py`](../backend/pupa_backend/credentials.py)) moves those vars out of
`os.environ` into an in-process dict at startup — the `claude` subprocess can't
inherit them, the guard passes honestly, and the deepagents model builders read
them via `get_credential`. Previously the two were mutually exclusive.

In this mode the **Claude Agent SDK** (`claude-agent-sdk`) runs in-process as
the sole tool-calling loop and **drives the iOS-forwarded frontend tools**:

- Each descriptor in `RunAgentInput.tools` becomes an in-process SDK MCP tool
  (`mcp__pupa_frontend__<name>`,
  [`frontend_tools.py`](../backend/pupa_backend/harnesses/claude/frontend_tools.py)). When Claude
  calls one, its handler parks on a future instead of executing.
- A per-thread **live-session registry**
  ([`registry.py`](../backend/pupa_backend/harnesses/claude/registry.py)) keeps the
  `ClaudeSDKClient` alive across the two HTTP requests of the AG-UI
  interrupt/resume contract. The first POST emits one batched
  `CustomEvent(on_interrupt)` + `RunFinished` and parks; the resume POST
  (`forwardedProps.command.resume.tool_results`, normalised by the shared
  `agui.tool_results.parse_tool_results`) resolves the futures and re-attaches
  a fresh SSE. **iOS is unchanged** — same wire shapes as `ag_ui_langgraph`,
  translated in [`events.py`](../backend/pupa_backend/harnesses/claude/events.py).
  Each `PendingCall` is tagged with the `run_id` that emitted it; `resolve_results`
  synthesises `missing_tool_result` only for unresolved calls of the batch the
  resume answers, so a resume for one run never clobbers a call still in flight
  from another. The parked wait is **per-tool** (`wait_timeout_for`) with
  env knobs (`PUPA_FRONTEND_WAIT_TIMEOUT` for CRUD,
  `PUPA_FRONTEND_WAIT_TIMEOUT_SLOW` for `invoke_agent`) — both default 300s and
  act as the absolute cap.
- **Client liveness heartbeat.** While a frontend tool is in flight
  the client POSTs `forwardedProps.command.keepalive {state}` every ~10s
  (interval-first, so sub-second CRUD dispatches never ping); the endpoint
  answers `204` and `session.keepalive()`s — no run starts. Once a client has
  pinged, `claim_call` waits only `last_keepalive + grace`
  (`PUPA_FRONTEND_LIVENESS_GRACE`, default 30s) instead of the full wall — a
  dead app fails in ~30s regardless of tool budget, while a 10-minute
  `invoke_agent` survives as long as pings arrive. `state: "background"` (sent
  on scene-phase background, iOS freezes timers) suspends the grace and falls
  back to the absolute wall so a briefly-backgrounded subagent isn't killed; a
  plain ping on foreground re-arms it. Clients that never ping keep the
  full-wall behaviour. A truly-abandoned turn relies on the wall or new-turn
  retirement (`registry.retire`), not a socket-close signal.
- **A re-attach counts as a ping.** `SSEReplayMiddleware` serves a re-attach by
  short-circuit, so it never reaches a loop — a returning app would otherwise
  stay marked `background` on a stale clock at the exact moment it proved it was
  alive. The middleware therefore fires `sse_replay.notify_reattach(thread_id)`,
  and the Claude loop registers `registry.note_reattach` as an observer at mount
  time. Observers are best-effort (a raising hook can't cost the client its
  replay tail) and registration is idempotent, since one loop mounts on both `/`
  and `/harnesses/{id}`. The callback direction keeps the harness boundary
  one-way: `sse_replay` never imports a harness.
- **One consumer per session queue.** A second POST can land on a live session —
  the client re-sends a resume whose response was lost, while the pump (which
  survives a client disconnect by design) is still draining. Two `attach()`
  generators on one `asyncio.Queue` each take a *share* of the events: the turn
  splits across two SSE responses and both append into the same `sse_replay` log
  through independent task chains, so the log's frame order stops matching the
  turn's. `attach()` therefore displaces the consumer already holding the
  session: it sets that attachment's `stop`, waits (bounded — a generator nobody
  is iterating must not stall the request replacing it) for its `done`, and the
  displaced generator hands back any event it had in flight via
  `LiveSession.pushback`, which the new consumer drains ahead of the queue so the
  handover can't reorder the turn.
- **Teardown always ends the run.** `LiveSession.dispose()` queues a terminal
  `RunError` + `ERROR` sentinel *before* it unblocks parked handlers, cancels
  the pump and disconnects the SDK client, so a session torn down mid-turn
  (thread reused by a fresh POST, idle sweep) can never leave an attached
  `attach()` drain blocked on its queue — otherwise the SSE ends with no
  `RunFinished`/`RunError` and a later re-attach reads an `sse_replay` log with
  `live=0`, i.e. a silent chat. If the pump already queued its own terminal that
  one is drained first. `registry.remove()` is identity-checked (`remove(tid,
  session)`) so a stale session's late drain never evicts the replacement that
  already claimed the thread.
- **A new turn retires the parked session first.** `dispose()` is the hard path:
  it cancels the pump and closes the transport at once, which rejects the CLI
  child's in-flight `hook_0` (PreToolUse) / permission roundtrip — the
  `Stream closed` / `Tool permission stream closed before response received`
  errors — and leaves the SDK session interrupted, so the next turn's `resume`
  is answered with a `Continue from where you left off.` no-op instead of the
  user's prompt. The new-turn POST therefore calls `registry.retire(thread_id)`
  before `create()`: release the parked handlers (`release_pending`), `interrupt()`
  the child, wait a bounded `PUPA_CLAUDE_RETIRE_DRAIN` (default 2s) for the pump
  to reach its terminal, then dispose. Every step is best-effort — the thread
  always ends up free for the newcomer. Matters most on app wake, where the
  returning app sends a fresh message on a thread parked mid-tool.
- **Assistant text streams token-by-token.** `ClaudeAgentOptions` sets
  `include_partial_messages=True`, so `receive_response()` yields partial
  `StreamEvent`s alongside the whole messages. `_pump` maps each text delta to an
  incremental `TextMessageContent` (`events.translate_stream_event`); the trailing
  whole `AssistantMessage` is translated with `skip_text` so the text isn't
  re-sent (tool calls, which don't stream, are still emitted from it). Parity with
  the deepagents harness — iOS accumulates deltas either way.
- **`skip_text` is per message, never blanket.** `translate_stream_event` records
  each message id that opened a text block; `events.text_already_streamed` decides
  `skip_text` from that set. The CLI fabricates some assistant messages locally —
  rate-limit notices, API errors, and the `No response requested.` reply it gives
  a `query()` queued behind a resumed session's `Continue from where you left
  off.` — and those never stream. Skipping them unconditionally deleted their
  text, leaving a run that emitted only `RUN_STARTED` + `RUN_FINISHED`, which the
  app renders as a dropped connection.
- **Native + server tool calls render display-only.** Claude's own in-process
  tools (`Read`/`Bash`/`Grep`/… and `ServerToolUseBlock` web_search / web_fetch)
  emit `ToolCallStart/Args/End` so the app shows a tool bubble, but are **not**
  added to `frontend_calls` — no `on_interrupt`, no pending slot, no dispatch,
  no result event (call-side only). Only `mcp__pupa_frontend__*` tools park for
  on-device dispatch. The guardrail: display-only calls must never enter
  `frontend_calls` ([`events.py`](../backend/pupa_backend/harnesses/claude/events.py)).
- **Same-turn tool unlock (continuation turn).** Frontend capabilities are gated
  behind `get_tools_<group>` activation tools; calling one makes the device
  advertise the group's real tools on its next request. An in-process SDK MCP
  server's tool list is **frozen at `connect()`** (the SDK advertises tools
  without `listChanged`, and the CLI *rejects* `mcp_toggle` for SDK servers), so
  the live client can't grow mid-session. When a resume POST advertises frontend
  tools the live client lacks (via `command.resume.tools_after_round`, iOS's
  post-dispatch snapshot, else `input.tools`), the endpoint delivers the gate
  result, **interrupts** the narrow turn (so it can't loop re-calling the gate for
  tools it can't reach), and the pump runs a **continuation turn**: a fresh
  `ClaudeSDKClient` exposing the widened surface, `resume`-ing the same SDK session
  with a synthetic "continue" prompt, streamed onto the same SSE. The loop system
  prompt tells the model to stop after an activation call rather than flail. Cost
  is one short model turn per activation. No explicit cap is needed: at most one
  continuation runs per resume POST (a continuation can't self-trigger), and a
  further activation needs another iOS resume POST, which iOS's own per-send round
  cap already bounds. iOS is unchanged.
- **Context continuity across failures.** The loop's only record of the
  conversation is the SDK session id it `resume`s (iOS re-sends *only* user
  messages). The pump remembers that id from the **earliest** message that carries
  it, not just the final `ResultMessage`, so a turn that errors or is interrupted
  before its result still leaves a resumable id — the next user turn keeps the
  prior assistant/tool context instead of starting blind.
- **Multimodal input.** A user message whose `content` is a list of AG-UI parts
  (`ImageInputContent` alongside text) is forwarded to the model as Anthropic
  content blocks: the endpoint streams a single structured user message to
  `ClaudeSDKClient.query()` (data sources → `image/base64`, url sources → `image/url`)
  instead of the plain-string path. Text-only turns still send a plain string.
  Without this, images were coerced to text-only and silently dropped.
- **Ambient context injection.** `RunAgentInput.context` (the frontend's
  per-turn `[{description, value}]` — live canvas state, memories snapshot, the
  MyApp system prompt / AGENTS.md) is rendered by `_render_context` and appended
  to the **end of the system prompt** (`_compose_system_prompt`, in
  `_options_for`). The loop builds its own prompt, so — unlike the
  `ag_ui_langgraph` harness, which folds context into the system prompt itself —
  the entries only reach the model if composed in here. System-prompt placement
  (not a per-turn user message) is deliberate: options are rebuilt on every
  new-turn/continuation POST and the SDK passes `--system-prompt` on each
  subprocess spawn (including `--resume`), so the volatile context refreshes *in
  place* each turn rather than accumulating a fresh copy in the transcript. It
  sits at the end so the stable base prompt prefix stays prompt-cacheable.
  Without this the loop dropped context entirely: the model never saw the app's
  instructions or canvas.
- Claude's **native** host tools are gated by a **PreToolUse hook**
  ([`gate.py`](../backend/pupa_backend/harnesses/claude/gate.py); `can_use_tool` is kept only as a
  backstop — the headless CLI skips it for auto-allowed tools, but the hook fires
  before every tool use): `PUPA_CLAUDE_LOOP_NATIVE` —
  `off` (cloud-pinned) blocks all; `read` allows Read/Grep/Glob (≈ Claude's
  *plan* mode, `permission_mode="plan"`); `edit` adds Edit/Write/Bash; `full`
  (alias `all`, **the default**) permits the **entire** native Claude Code toolset
  (web, subagents, …) so the agent can drive the host laptop itself. The loop is
  **permissive by default** — enabling the `claude_code` harness opts into
  Claude Code's power; cloud explicitly pins `off`. The app can switch the mode
  **per turn** via `state["claude_loop_native"]` (e.g. flip plan=`read` ↔
  edit=`full`) without a restart. The per-turn `state["disabled_tools"]` mute list
  is honoured for frontend tools too. Read-class (and, in `full`, read-only web) tools are
  pre-approved via `allowed_tools`; **mutating/command tools are deliberately not**
  (listing a tool in `allowed_tools` bypasses the gate), so they route through the
  hook and trigger the user-permission prompt. When native tools are enabled the
  loop system prompt gains a "Host machine tools" section so the model doesn't
  disclaim the host access it actually has.
- **Skills & MCP.** `PUPA_CLAUDE_LOOP_SKILLS` (`all` | comma-list | `off` default)
  enables installed Claude Code skills; when set, `setting_sources` loads
  `user`/`project` so the CLI can discover them (the PreToolUse hook still gates,
  so settings-level allow-rules don't bypass the permission prompt). The
  operator-configured MCP servers from `config.yml`'s `mcp_servers` block are the
  **same single shared connection** the deepagents path opens once at startup
  (`mcp_servers_lifecycle()`); the loop **bridges those already-connected tools
  in-process** as one SDK MCP server (`mcp__pupa_mcp__<tool>`,
  [`config_mcp.py`](../backend/pupa_backend/harnesses/claude/config_mcp.py)), so every thread reuses
  that one server instead of each `claude` subprocess spawning its own. This is
  deliberate: an external stdio/http server handed to the subprocess as
  `--mcp-config` is only loaded if the subprocess *trusts* it, but the loop runs
  with `setting_sources=[]` (no trust source) — so those tools never surfaced.
  In-process SDK servers need no trust (the same reason the frontend tools work),
  and the tool executes here in the backend process against the shared session.
  Their `mcp__*` tools are allowed without a per-call prompt (the operator opted in
  by configuring them). Unlike the deepagents path there is **no `get_tools` gate**
  yet — every configured MCP tool is exposed directly.
- **Surfacing Claude's interaction to the Pupa user.** The CLI's interactive
  built-ins (`AskUserQuestion`, `ExitPlanMode`) have no Pupa surface and are
  disallowed — `loop_system_prompt` tells the model to ask for clarification in
  **plain chat text** and stop, which the user answers with their next message
  (the loop resumes the same SDK session). And when an **edit-class native tool**
  wants to run, the gate parks and surfaces a yes/no **permission request** as
  chat text; the user's next message (`interpret_approval`) resolves it via the
  session's `pending_decision`. **The loop runs freely by default** (flow over
  friction) — the prompt only happens when approval is explicitly required via
  `claude_loop_require_approval: true` (`PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL=1`). Even
  then it's overridden by `claude_loop_auto_approve: true`, a per-turn
  `state["claude_loop_auto_approve"]` the app can toggle, or the user replying
  **"always"** once (sets `session.auto_approve` for the thread). The endpoint logs
  the resolved native scope + model at registration, and a loud warning when
  running permissively (host tools with no approval).
- **Model.** Selected per turn, same `forwardedProps.llm.model` channel the
  deepagents path uses ([`_resolve_loop_model`](../backend/pupa_backend/harnesses/claude/endpoint.py)):
  per-request wins, else `CLAUDE_CODE_MODEL` (config `claude_code_model`), else
  `claude-opus-4-8`. The loop is Claude-only, so the menu is claude-code's stable
  **aliases** — `opus`/`sonnet`/`haiku`/`fable`
  ([`models.py`](../backend/pupa_backend/harnesses/claude/models.py)), not pinned version
  ids (the `claude` CLI has no list-models command and aliases never go stale).
  `GET /harnesses` reports this alias menu under the `claude_code` harness (the
  deepagents harness reports its `MODEL_REGISTRY`). The `provider` field is ignored and a non-Claude
  pick (e.g. an OpenRouter slug) is rejected with a `RunErrorEvent` toast; a full
  `claude-*` id is accepted so a pinned config / stale client still works. The
  model the SDK reports is logged on the first assistant message and stored on the
  session (`sdk_model`).
- **Token / cache logging.** The pump logs usage in yellow at INFO
  ([`usage.py`](../backend/pupa_backend/harnesses/claude/usage.py)): one
  `claude_code tokens:` line per `AssistantMessage` — `in` / `out` /
  `cache_read` / `cache_write` plus the cache-hit share of the prompt, the only
  per-API-call view of the cache split — and one `claude_code turn totals:` line
  at the `ResultMessage` adding `total_cost_usd`, `num_turns`, API duration and
  the per-model `model_usage` breakdown. Totals are logged before the tool-unlock
  continuation hand-off, so a continued turn still reports what its predecessor
  burned. Keys are read tolerantly (`usage` is snake_case, `model_usage`
  camelCase); a missing `usage` skips the per-call line and degrades the totals
  line to `no token usage reported`. The per-call line is emitted once per
  message id — the SDK yields one `AssistantMessage` per content block and every
  one carries the same message-level usage.
- **Prompt-cache diagnosis.** Anthropic caches an exact `tools` → `system` →
  `messages` prefix and the `claude` CLI owns the `cache_control` breakpoints
  (`ClaudeAgentOptions` exposes no cache knob), so a turn that writes cache and
  never reads it means *our* prefix moved. `_options_for` rebuilds all three from
  scratch on every POST, so it fingerprints what it built — model, base system
  prompt, tool set / order / schemas, permission mode, thinking, skills, cwd, and
  each `input.context` entry's description and value separately — and logs which
  keys moved since the thread's previous turn:

  ```
  claude_code cache: prefix changed [ctx.live-canvas-state.value], tools=81 system=9,241ch — cache write expected (thread=…)
  claude_code cache: prefix unchanged, tools=81 system=9,241ch — cache read expected (thread=…)
  ```

  The key names mirror the CLI's own cache-miss reasons. Two structural costs are
  visible this way: a gate unlock widens the tool set, and tools are the *first*
  block, so the whole prefix re-writes (accepted — the alternative is advertising
  every tool up front); and the ambient context sits in the **system** prompt,
  which precedes `messages`, so any drift in it re-caches the entire transcript
  behind it — a cost that grows with conversation length. A run where the line
  says `prefix unchanged` but the tokens line still shows a large `cache_write`
  points at the messages block instead (transcript re-serialisation across the
  `resume`), not at the options build.
- **Ambient context is canonicalised.** `_context_pairs` re-serialises each
  context `value` that parses as a JSON object/array with `sort_keys=True`
  (`_canonical_json`; array order and non-JSON values untouched). The client
  builds these from Swift `Dictionary`s, whose iteration order is randomised, so
  an unchanged canvas/skills/type snapshot arrived with shuffled keys every turn
  — and because the block sits in the **system** prompt, ahead of `messages`, each
  reshuffle re-cached the whole transcript behind it (~11k of avoidable
  `cache_write` per turn, growing with conversation length). Normalising here
  covers every client, including shipped builds without the client-side
  `.sortedKeys`.
- **Prompt dump.** `PUPA_CLAUDE_PROMPT_DUMP=<dir>` (unset = off) writes the whole
  prefix per options build to `<dir>/<thread>/NNN.json` — tools, base system
  prompt, each ambient-context entry, composed system prompt, fingerprint — plus
  `NNN.diff`, a unified diff against the thread's previous turn (`(identical)`
  when nothing moved). Long text is stored as arrays of lines so the diff lands
  on the line that changed. **Off by default and not for a shared deploy**: the
  payload is user data at rest (canvas state, memories, AGENTS.md, every tool
  schema). Client-supplied thread ids are sanitised before use as directory
  names, and a failing dump is swallowed rather than costing the turn. See
  [`prompt_dump.py`](../backend/pupa_backend/harnesses/claude/prompt_dump.py).
- **Thinking.** Extended-thinking level is also selected per turn via
  `forwardedProps.llm.thinking`
  ([`resolve_thinking`](../backend/pupa_backend/harnesses/claude/thinking.py)):
  `auto` → `{type: adaptive}` (model decides), `off` → `{type: disabled}`,
  `low`/`medium`/`high` → `{type: enabled, budget_tokens: …}` (ascending budget),
  spread into `ClaudeAgentOptions.thinking`. No key / unknown value → option left
  unset (CLI default), so existing threads are unchanged. `GET /harnesses`
  reports the level menu under `thinking` for the `claude_code` harness
  (deepagents omits it → `[]`).

**Billing is subscription-only and fail-closed**
([`env.py`](../backend/pupa_backend/harnesses/claude/env.py)). The SDK wraps the `claude` CLI and
inherits its auth/billing resolution; because its subprocess **inherits the
parent env** (`options.env` only overlays, can't delete), and the CLI puts
`ANTHROPIC_API_KEY` ahead of the subscription token, the loop enforces by
**detect-and-refuse**: at registration it asserts no forbidden credential var is
present (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `AWS_*`, Bedrock/Vertex,
…) and probes `claude auth status --json`, requiring `loggedIn=true`,
`apiProvider=firstParty`, and `authMethod ∈ {claude.ai, oauth_token}`. Anything
else (api_key, third_party, none, unknown) raises `SubscriptionBillingUnavailable`
at startup rather than silently billing per-token API credits. There is no
api-billing fallback in this build.

**Caveats.** The live-session registry pins a thread to one backend instance, so
the loop is single-instance / self-hosted only (no horizontal scale without
sticky routing). Subscription ToS for automated/server use is the operator's
responsibility (same surface as the `claude_code` tool). Cloud
([`deploy/cloud-config.yml`](../deploy/cloud-config.yml)) stays on deepagents and
pins `claude_loop_native: "off"`.

## Tracing (Langfuse)

[`backend/pupa_backend/harnesses/langgraph/observability/tracing.py`](../backend/pupa_backend/harnesses/langgraph/observability/tracing.py) wires
opt-out Langfuse tracing into each AG-UI request. It runs automatically
once `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set — no enable
flag — and can be forced off with `PUPA_LANGFUSE_DISABLED=1`, which is an
absolute kill switch. Clients say nothing about Langfuse: the `trace_id`
comes from the AG-UI `run_id` (or a random UUID when that is not one) and
the `session_id` from the `thread_id`, so tracing configures itself from
identifiers each request already carries. Each round is one trace — view it
in Langfuse's **Traces** table (or filter `Is Root Observation = True`), not
the observation-level list.

**Version tracking.** Every trace is stamped with the build that produced
it:

- **Backend** — the lifespan sets `LANGFUSE_RELEASE=pupa-backend@<ver>`
  (via `os.environ.setdefault`, so a deploy override wins) using
  `backend_version()` (reads the installed `pupa-backend` version). This
  lands in the trace **Release** field. `build_langfuse_config` also adds
  a `backend:<ver>` tag and `backend_version` metadata.
- **Frontend** — not tracked. The backend cannot know the client build on
  its own, and teaching the client to send it would give the app a concept
  of Langfuse.

Only `langfuse_session_id` / `langfuse_tags` reach
the trace level through the LangChain `CallbackHandler` (langfuse v3); the
backend therefore carries version info via the `release` field and tags
rather than a per-trace `version`.

**Reading usage back.**
[`backend/pupa_backend/harnesses/langgraph/observability/usage.py`](../backend/pupa_backend/harnesses/langgraph/observability/usage.py) is the read
path companion. Because every trace's `session_id` *is* the `thread_id`,
a single Langfuse Metrics API call (view `traces`, summed `totalCost` +
`totalTokens`, grouped by `sessionId`) aggregates token + cost for many
threads at once. It is lazy/optional like the write path: when
`PUPA_LANGFUSE_DISABLED` is set, credentials are missing, or the query fails, it
returns `{}` and callers render `null` usage rather than erroring. Exposed
via `POST /db/threads/usage` (see `/db` routes below).

## Persistence

[`backend/pupa_backend/harnesses/langgraph/db/`](../backend/pupa_backend/harnesses/langgraph/db/)
abstracts checkpointer + store selection. It lives under the harness because
only the deepagents graph writes these checkpoints — the lifespan and the `/db` router are
both gated on `deepagents_harness_enabled()`.

The one shared piece is `DATABASE_URL` resolution
([`db_config.py`](../backend/pupa_backend/db_config.py)), which stays top-level
because [`auth/devices.py`](../backend/pupa_backend/auth/devices.py) reads it to
choose between the Postgres device store and the JSON file — auth must not
depend on a harness.

- **Config resolution** —
  [`db_config.py`](../backend/pupa_backend/db_config.py) reads a single
  `DATABASE_URL`. The URL *scheme* is the discriminator — no `db_type`
  key — and both the checkpointer and the store bind to that one URL.
  `postgres://` is normalised to the `postgresql://` psycopg requires.
  Unset → persistent SQLite under `~/.pupa-backend/`
  (`checkpoints.db` + `store.db`). The checkpointer uses
  `AsyncSqliteSaver` so chat history survives restarts; the store
  currently falls back to `InMemoryStore` because LangGraph has no
  async SQLite store implementation yet — `connection.py` logs a warning
  when it does this.
- **Hard requirement** — set `PUPA_REQUIRE_DB_SCHEME=postgresql` (or via
  YAML `persistence.require_db_scheme: postgresql`) to forbid the SQLite
  fallback. Startup fails loudly if no DB resolves or a different
  backend is picked. The cloud image pins this so multi-tenant deploys
  can never land on local SQLite (data dies with the container).
- **Lifespan ownership** —
  [`connection.py`](../backend/pupa_backend/harnesses/langgraph/db/connection.py)
  opens the underlying async connections inside the FastAPI lifespan and
  yields `(checkpointer, store)` to the graph builder. The store reaches
  the graph only; it is not an HTTP surface.
- **`/db` routes** —
  [`routes.py`](../backend/pupa_backend/harnesses/langgraph/db/routes.py) is four
  routes over the checkpointer, all `agent`-scoped:
  `GET /db/threads/{thread_id}/messages` normalises the latest checkpoint's
  messages into `TranscriptMessage` so the iOS client can reload old
  conversations, `DELETE /db/threads/{thread_id}` drops a thread's
  checkpoints, and the two usage routes below.
- **`POST /db/threads/usage`** (`agent` scope) — batched token + cost
  per `thread_id`, read from Langfuse via
  [`observability.usage.fetch_usage`](../backend/pupa_backend/harnesses/langgraph/observability/usage.py). Body is
  `{"thread_ids": [...]}`; response is `{thread_id: {total_tokens,
  cost_usd, input_tokens, output_tokens, fingerprint}}` from one Langfuse
  Metrics API call. Cached per thread keyed by the latest `checkpoint_id`
  (the `fingerprint`) with a 30s TTL grace, so Langfuse is re-queried only
  after a new turn or to cover ingestion lag. Powers the client's
  Agents-dashboard cost view (per-thread, per-agent, per-MyApp rollups).
  `null` totals when Langfuse is off.
- **`POST /db/threads/usage/cache`** (`agent` scope) — prompt-cache
  breakdown per thread (`input_total`, `input_cache_read`,
  `input_cache_creation`, `cache_read_pct`). The Metrics API can't sum the
  cache sub-keys, so [`observability.usage.fetch_cache`](../backend/pupa_backend/harnesses/langgraph/observability/usage.py)
  walks each session's generation observations — **heavier, on-demand**
  (the client calls it only when an agent row expands), same fingerprint +
  TTL cache and `null`-when-unavailable contract.

## Auth

[`backend/pupa_backend/auth/`](../backend/pupa_backend/auth/) implements pair-once auth.

- **Middleware** —
  [`middleware.py`](../backend/pupa_backend/auth/middleware.py) gates every request
  except a small allowlist (`/auth/config`, `/auth/pair`, and any
  `*/health`). It accepts either the bootstrap
  `PUPA_API_KEY` or a paired-device bearer token resolved via
  `DeviceStore.resolve`. `PUPA_AUTH_DISABLED=1` short-circuits the
  middleware — dev loops only, and `main()` enforces that: it **refuses to
  start** when the switch is set and the listener is reachable from off the
  machine, since a warning about a wide-open agent loop scrolls past in a
  platform log. Reachable means bound off-loopback *or* fronted by a tunnel —
  under `connectivity: cloudflared` the socket is on loopback and the URL is
  public, so the bind address alone would exempt the most exposed case.
  `PUPA_ALLOW_INSECURE_BIND=1` overrides it — a second, differently-named
  variable, so pasting the first into a launch script can't reach it.
- **Per-route authorization** —
  [`scopes.py`](../backend/pupa_backend/auth/scopes.py) provides
  `require_scope("<scope>")` and `require_api_key()` FastAPI
  dependencies on top of the middleware. The middleware sets
  `request.state.auth` to `("api_key", None)` or
  `("device", PairedDevice)`; the dependencies read it. `api_key`
  identity bypasses scope checks (operator god mode); a device must
  carry the named scope, else 403. Route map: `/db/threads/*`,
  `/harnesses`, and the run endpoints (`POST /`, `POST /harnesses/{id}`)
  → `agent` scope; `/auth/devices/*` and `/auth/pair/begin` →
  `require_api_key()` (operator-only). Minting is operator-only on
  purpose: a device that could mint a device would let a leaked token
  outlive the revocation of the device it was issued to.
- **Pairing** —
  [`pairing.py`](../backend/pupa_backend/auth/pairing.py) holds a short-lived
  `PairingCodeStore` of one-time 8-char codes minted by
  `/auth/pair/begin`. Code TTL defaults to 5 min (capped at 1 day);
  device-token TTL is also operator-configurable per-request.
- **Abuse limits** —
  [`ratelimit.py`](../backend/pupa_backend/auth/ratelimit.py) throttles the
  pairing routes per client (`/auth/pair` 5/min, `/auth/pair/begin` 10/min).
  **Only failed attempts are charged** — the charge goes on at entry and is
  refunded (by the exact timestamp it wrote, so it can't take back a concurrent
  request's) once the status code says the caller was legitimate; checking
  first and charging after would let requests arriving together all clear the
  check against a bucket none of them had written to. The cost of holding the
  budget for the request's duration: more than the limit genuinely in flight on
  one bucket 429s even when all would have succeeded — retrying works
  immediately, so `Retry-After` is an upper bound. A successful pairing is
  free, because
  on `/auth/pair` the code *is* the credential (one request, then it's
  consumed) and `/auth/pair/begin` is operator-only, so throttling a caller
  who already holds `PUPA_API_KEY` protects nothing. What it does throttle is
  wrong codes and wrong keys. Buckets are **per client with no global cap**: a
  shared bucket that blocks is a denial of service with extra steps, since a
  stranger could drain it and lock out everyone holding a real credential.
  Keyed on the **rightmost**
  `X-Forwarded-For` entry across *every* field line of that header — a proxy
  may append a second line rather than extend the caller's, and reading only
  the first would give the caller the last word — *when a proxy is trusted*,
  else on
  `request.client.host`. Both halves matter: the proxied modes terminate in
  front of a loopback listener, so the peer address is `127.0.0.1` for every
  remote caller and would bucket the internet as one — but on a direct bind
  the header is written by the caller, so believing it would let one host
  rotate it per request for unlimited buckets.
  Mounted *inside* `require_https_middleware`, so a plaintext hop is refused
  before it can spend a real device's budget — the limiter needs to know
  nothing about transport. Not `slowapi`: it charges before `call_next` and
  offers no post-response hook, so "charge only failures" cannot be expressed
  in it; the shared-bucket ceiling is an `OrderedDict` keyed by last charge, so
  eviction is O(1) and drops the bucket charged longest ago rather than a
  guesser still spending.
  `PUPA_RATE_LIMIT_DISABLED=1` opts out for local loops. `POST /` is
  deliberately *not* throttled — a dropped SSE socket re-attaches there, so a
  per-IP cap would break the flaky-network case `SSEReplayMiddleware` exists
  for.
- **Proxy trust** —
  [`proxy.py`](../backend/pupa_backend/auth/proxy.py) answers "should
  `X-Forwarded-*` be believed here", which every forwarded-header read depends
  on. `PUPA_TRUSTED_PROXY` (config `transport.trusted_proxy`) is explicit and
  wins **either way** — `transport.trusted_proxy: false` is written to the env
  as `0`, not omitted, so it overrides the inference below. Otherwise it's
  inferred true for `connectivity: tailscale` / `cloudflared`, since the
  backend starts those proxies itself; otherwise **false**. Pinned true in the
  cloud image for Railway. Wrong in the safe direction (real proxy, flag unset)
  collapses callers into one bucket; wrong the other way voids the rate limits
  and the HTTPS check, which is why the default is off.
  For this to be the only answer, uvicorn is started with
  `proxy_headers=False` ([`app.py`](../backend/pupa_backend/app.py)): its
  default middleware folds `X-Forwarded-Proto`/`-For` into the ASGI scope for
  any peer within `forwarded_allow_ips` (`127.0.0.1` — every tunnel mode, and
  anything else on the host), which would rewrite `url.scheme` and
  `client.host` *above* the app and make the check below read a forged value.
  Consequence for **operator-run** reverse proxies (nginx, Caddy, a manual
  `cloudflared`): set `transport.trusted_proxy: true`, or every caller buckets
  as `127.0.0.1` and `PUPA_REQUIRE_HTTPS` sees a plaintext hop.
  `main()` resolves the rest at startup, because `PUPA_CONNECTIVITY` says what
  was *intended* and the answer needed is what actually came up. Two separate
  facts:
  - **Fronted** — something local forwards into this listener, so it binds
    `127.0.0.1`. False when `tailscale serve` didn't start (CLI absent,
    `PUPA_TAILSCALE_SERVE=0`, or `serve` failed), which keeps the documented
    `0.0.0.0` fallback and logs why.
  - **Rewrites `X-Forwarded-*`** — that something is an HTTP proxy, so it
    overwrites what the caller sent. True for `cloudflared` and for Tailscale's
    **https** mode; false for Tailscale's **tcp** mode, which is a raw L4
    passthrough where the client's request arrives byte-for-byte and those
    headers stay caller-written. Only this sets `PUPA_TRUSTED_PROXY` when the
    operator hasn't.
  A non-loopback bind while forwarded headers are trusted (Railway, or an
  explicit flag) logs a warning: anyone who can reach the port directly can
  write their own hop.
- **Transport** —
  [`transport.py`](../backend/pupa_backend/auth/transport.py) implements
  `PUPA_REQUIRE_HTTPS` (config `transport.require_https`), unset by default so
  LAN and offline installs keep working. When set, any non-TLS request that
  isn't a health probe gets 403, and the screen-share WebSocket closes with
  4403 (WebSockets skip the HTTP middleware stack, so that check is inline).
  Refused sockets are accepted and *then* closed — a close on a socket that was
  never accepted has no frame to carry the code, so the client would see a bare
  HTTP 403 and couldn't tell 4403 from 4401.
  Secure means `url.scheme == https` **or** the rightmost `X-Forwarded-Proto`
  is `https`. There is deliberately no loopback carve-out, for the same reason
  the rate limiter can't key on the peer address. Pinned on in
  [`deploy/cloud-config.yml`](../deploy/cloud-config.yml).
- **Response headers** —
  [`headers.py`](../backend/pupa_backend/auth/headers.py) sets `nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and HSTS only on a
  connection that is actually TLS — which is `is_secure_request` and nothing
  else, so the tunnel modes get it too even though the cert lives in the
  terminator rather than here. No `CORSMiddleware`: there is no browser
  origin to allow, and a permissive one would let any web page call this
  backend with a user's credentials.
- **Device store** —
  [`devices.py`](../backend/pupa_backend/auth/devices.py) hashes and persists
  paired-device bearer tokens. `DeviceStore` writes a JSON file at
  `backend/pupa-auth.json` (override via `PUPA_AUTH_DB_PATH`).
  `PostgresDeviceStore` auto-selects when the checkpointer is Postgres,
  storing token hashes in a `pupa_devices` table — survives Railway's
  ephemeral filesystem.

## Screen-share broker

[`backend/pupa_backend/screenshare/`](../backend/pupa_backend/screenshare/) hosts the WebRTC
signalling broker mounted at `/screenshare/ws`. Opt in by setting
`PUPA_SCREENSHARE=1` (or via the YAML config).

- **WS auth** —
  [`routes.py`](../backend/pupa_backend/screenshare/routes.py) requires either a
  paired-device token with the `screenshare` scope, or a publisher
  connecting from loopback (`127.0.0.1` / `::1`) with the sidecar
  token. The sidecar token is generated on backend boot and written to
  `/tmp/pupa-sidecar.token` (configurable via
  [`sidecar_token.py`](../backend/pupa_backend/screenshare/sidecar_token.py)) — the
  Swift sidecar reads it without ever touching a paired-device token.
- **Broker** —
  [`broker.py`](../backend/pupa_backend/screenshare/broker.py) routes signalling
  JSON between exactly-one publisher and any viewers for a given
  `share_id`. Viewers may omit the share id if exactly one publisher
  is active. Publishers see a `4409` error if another publisher
  already owns their share id.

The macOS publisher lives outside `backend/` in
[`screenshare-sidecar/`](../screenshare-sidecar/) — see its
`Package.swift` and `Sources/PupaScreenshare/`.

## Tool gating

[`backend/pupa_backend/harnesses/langgraph/tool_gating.py`](../backend/pupa_backend/harnesses/langgraph/tool_gating.py) is a middleware
that drops tools the model shouldn't see for the current turn — for
example, the `tavily_search` tool is hidden when no `TAVILY_API_KEY` is
set. MCP server tools are gated separately by `McpGateMiddleware`
([`backend/pupa_backend/mcp_servers.py`](../backend/pupa_backend/mcp_servers.py)): each server's
tools stay hidden until the agent activates that server through the
`get_tools(server=…)` gate.

## Shell approval

[`backend/pupa_backend/harnesses/langgraph/shell_approval.py`](../backend/pupa_backend/harnesses/langgraph/shell_approval.py)
(`ShellApprovalMiddleware`) is a pause-before-execute gate installed
alongside the `shell` tool (always, whenever `SHELL_TOOL_ENABLED=1`).
Running shell commands unattended on the backend host is dangerous, so
the safe default is to ask the user before every execution.

- **Batched interrupt in `after_model`** — the approval `interrupt()`
  fires once per model turn (its own graph node), collecting every
  `shell` call in the turn that isn't already remembered and surfacing
  them as a single `request_shell_approval` `frontend_tool_calls`
  card — the same channel and pattern
  `CustomCopilotKitMiddleware` uses for frontend tools.
  Firing per-call inside `awrap_tool_call` (which `ToolNode` runs in
  parallel) instead raised multiple pending interrupts on a multi-`shell`
  turn and crashed on resume.
- **Execution gate in `awrap_tool_call`** — for each `shell` call the
  middleware reads the per-call decision `after_model` recorded in
  `state["shell_approval_decisions"]` and either runs the command
  (approved / pre-approved / disabled) or returns a denial `ToolMessage`
  without executing. A missing decision fails closed (denied).
- **Allow once / always** — the client resumes with
  `tool_results=[{toolCallId, content}]` where `content` is
  `{"approved": bool, "remember": bool}`. `remember=True` adds the exact
  command string to an in-memory per-thread allowlist (keyed by
  `thread_id`, lives on the middleware instance, cleared on restart), so
  the next identical command skips the interrupt.
- **Per-turn disable** — the iOS Settings toggle "Require shell approval"
  off sends `state["shell_approval_disabled"] = True`, and both hooks
  bypass the gate for that turn.

## Skills

[`backend/pupa_backend/harnesses/langgraph/skills.py`](../backend/pupa_backend/harnesses/langgraph/skills.py) gives the agent a skill
lifecycle modeled on deepagents' `SkillsMiddleware` (the Agent Skills
spec, agentskills.io), following the standard progressive-disclosure
shape. **On by default**; opt out with
`PUPA_SKILLS_DISABLED=1`. Wired through
[`backend_tools.py`](../backend/pupa_backend/harnesses/langgraph/backend_tools.py) like every other
backend capability. The cloud image pins it off (`skills_disabled: true`
in [`deploy/cloud-config.yml`](../deploy/cloud-config.yml)) until the
untrusted-skill mitigations land.

- **Location** — skills live under `~/.pupa-backend/skills/` (override
  with `PUPA_SKILLS_DIR`), next to the rest of the backend's state. The
  package ships **no** built-in skills; the directory is created empty on
  first run and populated by the user or, later, a marketplace install
  path. Each immediate subdirectory is one skill: a `SKILL.md` with YAML
  frontmatter (`name`, `description`, optional `metadata` /
  `allowed-tools`).
- **Progressive disclosure** — at agent start `PupaSkillsMiddleware`
  injects every skill's *name + description* into the system prompt
  (discovery). The agent loads a skill's full body only when it needs it,
  by calling the read-only `skill_view` tool (read). Bodies never sit in the
  per-turn payload until read — the same token discipline as the trimmed
  `write_todos` block in [`agent.py`](../backend/pupa_backend/harnesses/langgraph/agent.py).
- **Read surface** — deepagents' `FilesystemMiddleware` is deliberately
  *not* mounted: it would register seven read/write/execute tools every
  turn. Instead `skill_view` is a single scoped tool that reads
  `<skills>/<name>/SKILL.md`, validating `<name>` against traversal and
  separators so it cannot escape the skills directory.
- **Session caching** — `PupaSkillsMiddleware` inherits deepagents'
  `SkillsMiddleware` loading, which reads skills once per session
  and caches them in agent state, so a new or edited skill only appears
  in a fresh session (the iOS client mints one `thread_id` per New
  Session).

Not yet implemented: per-skill tool gating, a
LangGraph-store backend for per-device installed skills, frontend MyApp
parity (provenance labels + no-orphan session gating), and a marketplace
`install_skill` path.

## Config

[`backend/pupa_backend/pupa_config.py`](../backend/pupa_backend/pupa_config.py) loads the YAML
config at `~/.pupa-backend/config.yml` and translates it into env vars
at startup. Shell env always wins. The schema covers:

- `llm_providers.*` — named provider entries (`bedrock`, `anthropic`,
  `openai_compatible`, `openrouter`). `default_llm_provider` picks the
  active one; when omitted, the **first** entry (YAML document order) is
  used. The config file is the sole source of the default — `make backend`
  no longer injects one, so a shell-exported `LLM_PROVIDER` still wins but
  there is no hardcoded fallback (a `default_llm_provider` naming a missing
  entry leaves `LLM_PROVIDER` unset and startup fails loudly).
  `openrouter` entries carry just `model` (→ `LLM_MODEL`) and an optional
  `api_key` (→ `OPENROUTER_API_KEY`, else from shell).
- `env.*` — arbitrary env vars, passed through verbatim. The escape hatch
  for anything the typed schema doesn't name (raw AWS keys, an MCP server's
  token, a third-party SDK's var). Applied *before* the typed keys, so a
  documented key wins on collision. It matters most for the OS service, which
  inherits nothing from the operator's shell: without it, a var outside the
  schema could only ever be set by exporting it in a terminal, and so could
  never reach a background service.
- `service.check_env` — extra var names for the `service-install` guard to
  watch (see below). `known_env_vars()` already covers the whole typed schema;
  this list adds the `env.*` ones, which have no schema entry to derive from.
- `auth.api_key` — optional bootstrap key (the env var
  `PUPA_API_KEY` still wins).
- `persistence.database_url` / `persistence.require_db_scheme` —
  checkpointer + store config (env wins via `DATABASE_URL` /
  `PUPA_REQUIRE_DB_SCHEME`).
- `tls.cert` / `tls.key` — paths to a self-signed cert for local
  HTTPS. When set, `app.py` starts uvicorn with TLS. The cert `setup` mints is
  **397 days** and carries the `serverAuth` EKU: Apple clients refuse a TLS
  server cert valid for more than 398 days or missing that EKU, and they refuse
  it *before* the app's fingerprint pinning runs, so the failure surfaces as a
  generic "refused a secure connection". Startup warns when the configured cert
  breaks those rules or is near expiry; renewing means re-running
  `pupa-backend setup` and re-pairing (new cert ⇒ new pinned fingerprint).
- `connectivity` (`tailscale` / `cloudflared` / `localhost`) — how the
  phone reaches the backend (→ `PUPA_CONNECTIVITY`). With `cloudflared`,
  `app.py` starts a tunnel at boot as a managed child process (terminated
  on shutdown): a **named** tunnel if one is configured, else a quick tunnel.
  With `tailscale`, `app.py` binds **`127.0.0.1`** and registers a
  `tailscale serve` forward at boot
  ([`tailscale_proxy.py`](../backend/pupa_backend/tailscale_proxy.py)),
  removing it on shutdown. This exists because macOS gates connections to
  `0.0.0.0`-bound sockets behind the Local Network privacy grant; without it
  *nothing* reaches the backend, including `pupa-backend pair` on the same
  machine. Two modes, chosen by whether the tailnet has HTTPS enabled
  (`tailscale status --json` → `CertDomains`):
  - **https** — `serve --https=443 http://127.0.0.1:<port>`. tailscaled
    terminates TLS with an auto-renewing Let's Encrypt cert for the MagicDNS
    name; the backend serves plain HTTP and `pupa-backend pair` publishes
    `https://<magicdns>` with **no fingerprint**. Preferred: iOS refuses
    self-signed certs before the client's pinning delegate runs.
  - **tcp** — `serve --tcp=<port> tcp://127.0.0.1:<port>`, raw passthrough.
    The backend serves its self-signed cert end-to-end and the client pins the
    fingerprint.

  `PUPA_TAILSCALE_SERVE=0` opts out; `=tcp` / `=https` forces a mode;
  `PUPA_HOST` overrides the bind address.
- `cloudflared.hostname` / `cloudflared.tunnel` — set only for a Cloudflare
  **named** tunnel on the operator's own domain (→ `PUPA_CLOUDFLARED_HOSTNAME`
  / `PUPA_CLOUDFLARED_TUNNEL`). The hostname gives a **stable** public URL
  (e.g. `https://api.yourdomain.com`), so `make pair` / the QR use it directly.
  `pupa-backend run` spawns `cloudflared tunnel run <name>` itself (so the
  backend service brings the tunnel up too); `make tunnel-named` runs it
  standalone.
- `shell_tool_enabled`, `tavily_api_key`, `langfuse.*` — feature gates
  and external service creds.
- `claude_code_disabled` (opt-out), `claude_code_model`,
  `claude_code_workspace` — the `claude_code` tool's gate and config.
- `harnesses` — nested block of enabled [agent harnesses](#agent-harnesses-multiple-mounted-together),
  e.g. `{deepagents: {enabled: true, default: true}, claude_code: {enabled: true,
  native: "full"}}`. Serialised to `PUPA_HARNESSES` (JSON); the Claude harness's
  nested knobs (`native`/`skills`/`auto_approve`/…) also flatten onto the legacy
  `PUPA_CLAUDE_LOOP_*` env vars. Replaces the retired single `agent_loop:` switch.
  String values → quote them in YAML.
- `mcp_servers` — structured block of named MCP servers attached to the
  agent ([`backend/pupa_backend/mcp_servers.py`](../backend/pupa_backend/mcp_servers.py)); serialised
  to `PUPA_MCP_SERVERS` (JSON). Like `llm_providers`, it's a nested block,
  not a flat key.

The setup wizard ([`backend/pupa_backend/scripts/setup.py`](../backend/pupa_backend/scripts/setup.py),
`pupa-backend setup`) writes this YAML interactively. Its first questions
**enable each harness** and pick the default: enabling `deepagents` prompts for
the LLM providers as usual; enabling `claude_code` runs a soft `claude auth
status` preflight (warning if no first-party subscription login is found). Both
can be enabled together — the credential stash keeps them compatible. The **connectivity** question offers a full-auto
Cloudflare *named* tunnel: pick `cloudflared` + "I have a domain" and the wizard
creates the tunnel, routes DNS, and writes `~/.cloudflared/config.yml` for a
stable URL on the operator's domain (falling back to the quick tunnel if
`cloudflared` is missing or not logged in).

The cloud image at [`deploy/cloud-config.yml`](../deploy/cloud-config.yml)
uses the same schema. It bakes in the multi-tenant safety posture
(`shell_tool_enabled: false`, `screenshare: false`, no `mcp_servers` block),
sets `default_llm_provider: anthropic`, and lists both `anthropic` and
`bedrock` under `llm_providers` so an operator can switch by overriding
`LLM_PROVIDER` on Railway. Secrets always come from Railway env vars,
never from the YAML.
