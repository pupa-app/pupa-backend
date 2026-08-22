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
- email **support@pupa-app.com**.

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
  bearer token; the operator `PUPA_API_KEY` never reaches a client. Per-route
  authorization is enforced via scopes (see
  [`backend/pupa_backend/auth/scopes.py`](backend/pupa_backend/auth/scopes.py)),
  including the agent run endpoints.
  `PUPA_AUTH_DISABLED=1` removes all auth and is **only** for a same-machine dev
  loop — never expose such a backend to a network.
- **Devices cannot mint devices.** `/auth/pair/begin` requires `PUPA_API_KEY`;
  a paired-device token gets 403. So a leaked token cannot issue itself a
  replacement, and revoking the device it belongs to actually ends its access.
  Keep `PUPA_API_KEY` set — it is the only credential that can pair.
- **Failed pairing attempts are rate limited.** `/auth/pair` is the one
  unauthenticated write route (the bootstrap code *is* the credential), so
  wrong codes are throttled per client — as are wrong `PUPA_API_KEY` values on
  `/auth/pair/begin`. Successful requests are never charged, and buckets are
  per-client with no shared cap, so no amount of third-party abuse can block a
  caller holding a valid credential. Bucketing uses `X-Forwarded-For`, because
  every supported transport terminates TLS in front of a loopback-bound
  listener.
- **TLS is the operator's to enforce.** `PUPA_REQUIRE_HTTPS=1` refuses
  plaintext; it is pinned on in the cloud image and **must** be set on any
  internet-reachable self-host. See [docs/deploy.md](docs/deploy.md).
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
