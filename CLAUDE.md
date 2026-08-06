# Pupa backend — agent guide

**Never** add an AI signature or co-author trailer to any commit, PR, or issue.
**Never** leave personal information in the repo — this includes developers'
account names, real names, email addresses, and absolute paths from their
machines.

FastAPI backend for the Pupa client. AG-UI over a single `POST /` SSE stream,
served by one of two interchangeable agent loops; pair-once auth; optional
macOS screen-share sidecar that publishes WebRTC video into the
`/screenshare/ws` broker on this backend.

## Read these first

- [docs/architecture.md](docs/architecture.md) — backend shape **right now**.
  Source of truth.
- [docs/deploy.md](docs/deploy.md) — Railway + Postgres + Langfuse runbook.
- [backend/pupa_backend/harnesses/](backend/pupa_backend/harnesses/) — the
  registry plus one subpackage per agent loop.
- [screenshare-sidecar/](screenshare-sidecar/) — optional macOS Swift CLI
  publisher (WebRTC).
- [CHANGELOG.md](CHANGELOG.md) — patch-only bumps.

## Layout traps

- **The harness is called `deepagents`; its directory is `langgraph/`.** The id,
  label, config key, and route (`POST /harnesses/deepagents`) all say
  *deepagents*; the code lives in
  [`harnesses/langgraph/`](backend/pupa_backend/harnesses/langgraph/). Don't
  "fix" either one to match the other — the directory name is about the library,
  the id is about the loop.
- **Nothing above the harness boundary may import from a harness.** Shared code
  goes in [`agui/`](backend/pupa_backend/agui/) (the resume-payload parser both
  loops use) or top-level modules.
  [`db_config.py`](backend/pupa_backend/db_config.py) is top-level precisely
  because [`auth/devices.py`](backend/pupa_backend/auth/devices.py) reads it to
  pick a device store, and auth must not depend on a harness.
- **`/db` and the checkpointer lifespan are gated on the deepagents harness.**
  Both read and write LangGraph checkpoints, so a Claude-only deploy opens no
  database and mounts no `/db`.

## When changing behaviour — update the docs

On any change (new tool, route, persistence rule, auth flow, prompt,
dependency): update [docs/architecture.md](docs/architecture.md) to reflect
reality. Add a [CHANGELOG.md](CHANGELOG.md) entry under the next patch bump for
user-visible changes.

## Versioning

`0.0.X` versions. **Bump patch only** (`0.0.1` → `0.0.2`) unless the user says
otherwise. Never bump minor or major without instruction.

| Package | Version file |
|---|---|
| Python backend | `backend/pyproject.toml` |
| Screen-share sidecar | `screenshare-sidecar/Sources/PupaScreenshare/Version.swift` (`PupaScreenshareVersion`) |

Bump a sub-package version when its code changes; bump the root project version
(CHANGELOG + README table) when shipping any release-worthy change.

## Branching & releases

See [CONTRIBUTING.md](CONTRIBUTING.md). Branch from `dev`, squash-merge to
`dev`, fast-forward `main` from `dev` for releases. A push to `main` publishes
to PyPI automatically when `backend/pyproject.toml`'s version is new — PyPI
Trusted Publishing matches on repo + the `publish.yml` **filename** + the `pypi`
environment, so renaming any of those three breaks releases.

## Conventions

- **Tools are the client's by default.** The backend forwards frontend tools'
  JSON-Schema definitions to the model and the *client* executes them. Don't
  enumerate frontend tools in the system prompt — names, schemas, and
  descriptions are already forwarded as proper tool definitions, and
  duplicating them causes drift. See
  [`prompts.py`](backend/pupa_backend/prompts.py).
- **Backend tools are the short list**, all in
  [`harnesses/langgraph/backend_tools.py`](backend/pupa_backend/harnesses/langgraph/backend_tools.py):
  `tavily_search` (needs `TAVILY_API_KEY`), an env-gated `shell` behind an
  approval middleware, `write_todos`, the `claude_code` delegation tool,
  `skill_view`, plus any configured MCP server's tools behind a `get_tools`
  gate. Everything else is a frontend tool.
- **Both harnesses share one system prompt.**
  [`prompts.py`](backend/pupa_backend/prompts.py) is above the harness
  boundary; the Claude loop appends its own suffixes at request time in
  [`claude/env.py`](backend/pupa_backend/harnesses/claude/env.py). Editing the
  base prompt changes both loops.
- **The credential stash is load-bearing.** When the `claude_code` harness is
  enabled, [`credentials.py`](backend/pupa_backend/credentials.py) moves
  `ANTHROPIC_API_KEY` / `AWS_*` out of `os.environ` at import time so the
  `claude` subprocess can't inherit them and divert billing off the
  subscription. Read those vars via `get_credential`, never `os.getenv`.
- **SSE middleware order is load-bearing.** In
  [`app.py`](backend/pupa_backend/app.py) replay is added *first* so it sits
  innermost, under the keep-alive. Reordering puts heartbeat comments into the
  replay log.
- **Persistence env var.** A single `DATABASE_URL` drives
  [`db.open_persistence`](backend/pupa_backend/harnesses/langgraph/db/connection.py)
  — the URL *scheme* selects the backend, so there is no separate `db_type`
  key. Both the checkpointer and the store bind to it. Unset → persistent
  SQLite under `~/.pupa-backend/`. Cloud deploys pin
  `PUPA_REQUIRE_DB_SCHEME=postgresql` (via `persistence.require_db_scheme` in
  [`deploy/cloud-config.yml`](deploy/cloud-config.yml)) to forbid that fallback
  and fail fast when `DATABASE_URL` is missing.
- **Auth model.** Auth is required by default. The only client credential is
  the paired-device token in the iOS Keychain. Bootstrap: set `PUPA_API_KEY`
  server-side, run `make pair` → 8-char code, paste / scan in iOS Settings →
  Backend → Edit. The operator can `unset PUPA_API_KEY` after the first pair —
  paired devices keep working, fresh devices get 401 until the next pair. The
  key never reaches a client. `PUPA_AUTH_DISABLED=1` disables everything — dev
  loops only, never on a reachable backend. Per-route authorization lives in
  [`auth/scopes.py`](backend/pupa_backend/auth/scopes.py): scope-gated surfaces
  (`/db/threads/*` and `GET /harnesses` → `agent`) and operator-only surfaces
  (`/auth/devices/*`). API-key identity bypasses scope checks; devices must
  hold the named scope.
- **Write a test early when helpful.** For bugs / features, prefer a failing
  test up front; skip only for prompt copy and docs. Suite lives in
  [`backend/tests/`](backend/tests/) (pytest, `asyncio_mode = auto`).

## Run

`make help` for targets. `make backend` → FastAPI on `:8004`; `make pair` mints
a pairing code; `make screenshare` runs the macOS sidecar against the local
broker. AWS Bedrock or Anthropic creds required before `make backend` — set
them in your shell. See [`.env.example`](.env.example).
