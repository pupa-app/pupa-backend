# Security Policy

## Supported versions

pupa-backend ships patch-only `0.0.X` releases and follows the latest published
version. Security fixes land on the latest release; please upgrade to the newest
`pupa-backend` before reporting.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately via either channel:

- GitHub's [private vulnerability reporting](https://github.com/pupa-app/pupa-backend/security/advisories/new)
  ("Report a vulnerability" under the repository's **Security** tab), or
- email **pupa-app-help@proton.me**.

We aim to acknowledge a report within a few days and will coordinate a fix and
disclosure timeline with you.

When reporting, please include:

- affected version (`pupa-backend --version` / `GET /auth/config`),
- a description of the issue and its impact,
- steps to reproduce or a proof of concept, if available.

## Scope and hardening notes

pupa-backend is designed to be **operator-run**. A few properties are important
when assessing risk:

- **Auth is required by default.** The only client credential is a paired-device
  bearer token; the bootstrap `PUPA_API_KEY` never reaches a client. Per-route
  authorization is enforced via scopes (see
  [`backend/pupa_backend/auth/scopes.py`](backend/pupa_backend/auth/scopes.py)).
  `PUPA_AUTH_DISABLED=1` removes all auth and is **only** for a same-machine dev
  loop — never expose such a backend to a network.
- **Secrets are environment-driven.** LLM/provider credentials live in the
  operator's shell environment, never in the repo or the config file. The paired
  device token store holds only hashed tokens and is kept outside the repo
  (`~/.pupa-backend/`).
- **Powerful optional features are off by default** (shell tool, MCP servers,
  screen-share). The cloud image additionally pins them off. Enable them only on
  trusted hosts.

Please flag any finding that weakens these properties (auth bypass, token
leakage, scope escalation, SSRF via tools, etc.) through the private channel
above.
