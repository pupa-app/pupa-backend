<p align="center">
  <img src="docs/assets/pupa-icon.png" alt="Pupa" width="280" />
</p>

# Pupa backend

[![pupa-backend](https://img.shields.io/badge/backend-0.0.87-3776ab?logo=python&logoColor=white)](backend/pyproject.toml)
[![screenshare](https://img.shields.io/badge/screenshare-0.0.6-f05138?logo=swift&logoColor=white)](screenshare-sidecar/Sources/PupaScreenshare/Version.swift)

Your own server for the [Pupa](https://pupa-app.com) app. Install it on your
laptop, or in the cloud, pair your phone **once**, and the agent runs on your
machine, with your files, under your control.

Two agents are built in: **Claude Code**, billed to your Claude subscription, or
a **model of your choice** from any provider you have an API key for. Pick one
when you connect.

## Contents

- [Install](#install)
- [Connect the app](#connect-the-app)
- [Pairing and access](#pairing-and-access)
- [Pick your agent](#pick-your-agent)
- [Optional extras](#optional-extras)
- [Screen sharing](#screen-sharing)
- [Run it in the cloud](#run-it-in-the-cloud)
- [Contributing](#contributing)
- [Built on](#built-on)

## Install

Install from PyPI with [uv](https://docs.astral.sh/uv/). This puts the
`pupa-backend` command on your PATH:

```bash
uv tool install pupa-backend            # latest
uv tool install pupa-backend==0.0.72    # or pin a version your app is happy with
```

Or use the one-line installer, which sets up `uv` for you and runs the setup
wizard:

```bash
curl -fsSL https://raw.githubusercontent.com/pupa-app/pupa-backend/main/install.sh | bash
```

Then:

```bash
pupa-backend setup            # asks a few questions, writes ~/.pupa-backend/config.yml
pupa-backend run              # start it, on port 8004
pupa-backend pair             # show a QR code to scan with your phone
```

Handy afterwards:

```bash
pupa-backend status           # is it running?
pupa-backend service-install  # keep it running in the background, restart on boot
pupa-backend logs             # follow the log
pupa-backend stop             # stop it
```

Settings live in `~/.pupa-backend/config.yml`, written by the wizard and safe
to edit by hand. API keys for your model provider stay in your shell environment,
not in that file; see [`.env.example`](.env.example).

## Connect the app

Your phone reaches the backend over whatever network path suits you. Pairing
works out the right address for that path and puts it in the QR code, so the app
connects straight away.

<p align="center">
  <img src="docs/assets/backend-flow.svg" alt="Three ways the Pupa app reaches your backend: localhost on the same machine, a Cloudflare tunnel, or Tailscale." width="720" />
</p>

- **Same network**: phone and backend on the same Wi-Fi, or the app and backend
  on the same Mac. Nothing to set up; this is the default.
- **[Tailscale](https://tailscale.com)**: a private network across your own
  devices, so you can reach the backend from anywhere without opening a port.
  Set `connectivity: tailscale` in your config.
- **[Cloudflare tunnel](https://www.cloudflare.com/products/tunnel/)**: gives
  your backend an HTTPS address on the public internet, still without opening a
  port. Set `connectivity: cloudflared` and `pupa-backend run` starts the tunnel
  for you. A *quick tunnel* needs no signup but changes address on every restart
  (re-pair each time); a *named tunnel* keeps a stable address on your own
  domain, set up once in `pupa-backend setup`.

For a backend you deployed somewhere else, name it directly:
`pupa-backend pair --public-url https://<your-host>`.

## Pairing and access

The backend is closed by default. The only thing that gets a device in is a
token stored in the app's Keychain when you pair it.

1. Set `PUPA_API_KEY` in the backend's environment. This is the operator key,
   and it never leaves your server.
2. Run `pupa-backend pair` for an 8-character code and QR.
3. In the app: **Settings → Backend → Edit**, then scan or paste.

Only someone holding `PUPA_API_KEY` can pair a new device: a paired phone
can't pair others. Keep the key set if you want to add devices later; devices
already paired keep working either way, and you can revoke any of them.

## Pick your agent

Everything around the agent is the same whichever one you choose: the same app,
the same pairing, and most of all **your app's own tools**. The app
tells the backend what it can do, the agent decides when to use it, and the work
happens on your device.

What differs is who answers and how it's billed.

### Claude Code, what most people will want

Runs the full Claude Code agent inside the backend. Billed to **your Claude
subscription**, never per token. It refuses to start if per-token API keys are
lying around, so it can't quietly charge you twice. You get Claude's own
abilities (reading files, running commands, searching the web) shown live in the
app, plus your app's tools. Nothing to configure beyond being logged into the
`claude` command.

### Bring your own model

The other agent talks to **any provider you have a key for** (AWS Bedrock,
Anthropic, or anything OpenAI-compatible) and is billed per token by that
provider. It can swap models per request, keeps its own conversation history,
and adds a few server-side tools of its own: web search, a shell tool, delegation
to sub-agents, and skills. Reach for it when you want a specific model, or don't
have a Claude subscription.

| | **Claude Code** | **Your own model** |
|---|---|---|
| Billing | Your Claude subscription | Per token, by your provider |
| Setup | Log into the `claude` command | A provider API key |
| Models | Claude | Any model, switchable per chat |
| Built-in tools | Claude's own (files, shell, web) | Web search, shell, sub-agents, skills |
| Your app's tools | ✅ | ✅ |
| MCP servers | ✅ | ✅ |

Turn either on in `config.yml`'s `harnesses:` block. Enable both and the app
lets you choose per connection. Anything else, like MCP servers, tracing and screen
sharing, works the same with both.

These two are just the ones that ship. The agent slot is pluggable, and a third
can be added without touching the app: see
[CONTRIBUTING.md](CONTRIBUTING.md#adding-an-agent) if you want to write one.

## Optional extras

Add these as keys in `~/.pupa-backend/config.yml`, then restart. Each one also
has an environment variable, handy for containers and one-off runs. The
environment wins when both are set.

| Feature | In `config.yml` | Environment | What it does |
|---|---|---|---|
| **Web search** | `tavily_api_key: tvly-…` | `TAVILY_API_KEY` | Lets your own model search the web. Get a key at [app.tavily.com](https://app.tavily.com). Without it the agent answers from what it already knows. |
| **MCP servers** | `mcp_servers:` (e.g. `pupa-backend mcp add --playwright`) | `PUPA_MCP_SERVERS` | Plug in any MCP server (browser control, issue trackers, your own). Shared by both agents. Playwright also needs Node.js. |
| **Screen sharing** (macOS) | `screenshare: true` | `PUPA_SCREENSHARE` | Lets the app watch a window on your Mac. See [below](#screen-sharing). |
| **Shell access** | `shell_tool_enabled: true` | `SHELL_TOOL_ENABLED` | Lets your own model run commands on this machine, asking you first each time. Machines you trust only. |
| **Tracing** | `langfuse:` with `public_key` / `secret_key` | `LANGFUSE_PUBLIC_KEY` · `LANGFUSE_SECRET_KEY` · `PUPA_LANGFUSE_DISABLED=1` | Records every request to [Langfuse](https://langfuse.com) so you can see what the agent did. On as soon as keys are present. |
| **Sub-agents** | on by default | `PUPA_SUBAGENTS_DISABLED=1` | Lets your own model hand parts of a job to specialist helpers. |
| **Skills** | on by default, `skills_disabled: true` to turn off | `PUPA_SKILLS_DISABLED=1` | Reusable instructions you drop in `~/.pupa-backend/skills/`, loaded only when relevant. Starts empty. |

Anything not in that table can go under an `env:` block, which is passed through
to the backend as-is:

```yaml
env:
  SOME_SDK_TOKEN: ...
```

That block is the only way to give a setting to the background service
(`pupa-backend service-install`), which — unlike `pupa-backend run` — does not
inherit your shell. If you keep a key in `~/.zshrc`, the service can't see it,
and install will stop and tell you so rather than leave you with a service that
won't start.

Conversation history is kept for you automatically: on your machine by default,
or in Postgres if you point `DATABASE_URL` at one.

## Screen sharing

An optional macOS helper shares a window with the app, so the agent can see what
you're looking at:

```bash
pupa-backend screenshare
```

It's a small Swift program built from source, so it isn't part of the Python
install. The command tells you how to get it if it's missing. macOS 14 or later.

## Run it in the cloud

[`docs/deploy.md`](docs/deploy.md) walks through a hosted deployment (Railway +
Postgres), with the safer defaults a shared server wants: no shell access, no
screen sharing.

## Contributing

Pull requests welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the workflow,
running from a clone, and how to add your own agent.
[docs/architecture.md](docs/architecture.md) is the full technical reference.

## Built on

Pupa backend stands on open-source shoulders. The keystone is the
**[AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui)**, the one language
the app and this backend share. Everything else plugs into it:

| Project | Role here |
|---|---|
| **[AG-UI](https://github.com/ag-ui-protocol/ag-ui)** | **The key dependency**: the agent ↔ app event protocol. |
| **[Claude Code](https://github.com/anthropics/claude-code)** + **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)** | The Claude Code agent, billed to your subscription. |
| **[CopilotKit](https://github.com/CopilotKit/CopilotKit)** | Bridges the graph-based agent to AG-UI. |
| **[deepagents](https://github.com/langchain-ai/deepagents)** | The bring-your-own-model agent: sub-agents, skills, files. |
| **[LangGraph](https://github.com/langchain-ai/langgraph)** / **[LangChain](https://github.com/langchain-ai/langchain)** | The agent loop and its saved history underneath it. |
| **[Langfuse](https://github.com/langfuse/langfuse)** | Tracing and observability. |
| **[FastAPI](https://github.com/fastapi/fastapi)** + **[Uvicorn](https://github.com/encode/uvicorn)** | The web server. |
| **[Tavily](https://github.com/tavily-ai/tavily-python)** | Optional web search. |
