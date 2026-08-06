# pupa-backend

FastAPI / LangGraph backend that powers the [Pupa](https://github.com/pupa-app/pupa-backend)
iOS / macOS client. Speaks plain [AG-UI](https://github.com/ag-ui-protocol/ag-ui)
over a single `POST /` SSE stream; pair-once auth keeps a long-lived token in
each device's Keychain. Runs locally next to the client, or in the cloud
(Railway + Postgres).

## Install

Install from PyPI with [uv](https://docs.astral.sh/uv/) (isolated tool venv, puts
the `pupa-backend` CLI on your PATH):

```bash
uv tool install "pupa-backend[setup]"          # latest
uv tool install "pupa-backend[setup]==0.0.72"  # pin to match your app
```

## Quick start

```bash
pupa-backend run              # start the backend on :8004
pupa-backend pair             # mint a QR pairing code for your iPhone
pupa-backend status           # is it running?
pupa-backend service-install  # run as launchd / systemd background service
```

LLM credentials (AWS Bedrock, Anthropic, or any OpenAI-compatible endpoint) live
in your shell environment, never in the config file. Configuration is written by
`pupa-backend setup` to `~/.pupa-backend/config.yml`.

## Auth model

The backend **requires auth by default** — the only client credential is a
paired-device token in the iOS Keychain. Set `PUPA_API_KEY` server-side, run
`pupa-backend pair` for an 8-char code, and pair from the app. For a same-laptop
dev loop only, `PUPA_AUTH_DISABLED=1` skips auth entirely (never on a reachable
backend).

## Documentation

Full docs, architecture, deploy runbook, and contributing guide live in the
[GitHub repository](https://github.com/pupa-app/pupa-backend):

- [Architecture](https://github.com/pupa-app/pupa-backend/blob/main/docs/architecture.md)
- [Deploy (Railway + Postgres + Langfuse)](https://github.com/pupa-app/pupa-backend/blob/main/docs/deploy.md)
- [Contributing](https://github.com/pupa-app/pupa-backend/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/pupa-app/pupa-backend/blob/main/CHANGELOG.md)

## License

[MIT](https://github.com/pupa-app/pupa-backend/blob/main/LICENSE)
