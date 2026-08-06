"""In-process credential stash so the LangGraph and Claude Code harnesses can
coexist in one server.

## Why this exists

The Claude Code harness drives the `claude` CLI (the Agent SDK spawns the
binary), which inherits the parent `os.environ`. Its billing guard
(`claude_loop/env.py`) is subscription-only and **refuses to start** if any
billing-diverting credential var (`ANTHROPIC_API_KEY`, `AWS_*`, …) is present in
the parent env — the SDK can't strip an inherited var, so a stray key would
silently bill per-token API credits.

But the deepagents harness *needs* exactly those vars. To run both harnesses in
one process we move the credentials **out of `os.environ`** into this private
in-process dict at startup: the LangGraph model builders read them via
`get_credential`, while the `claude` subprocess (which only ever sees
`os.environ`) can no longer inherit them. The Claude guard's
`assert_no_forbidden_env()` then passes honestly — the invariant it enforces
(the subprocess never inherits a diverting credential) genuinely holds.

Scrubbing runs **only** when the Claude Code harness is enabled; otherwise the
vars stay in `os.environ` untouched and `get_credential` simply reads through to
`os.getenv`.
"""

from __future__ import annotations

import logging
import os

from pupa_backend.harnesses.claude.env import FORBIDDEN_ENV_VARS

logger = logging.getLogger("uvicorn.error")

# Populated by `stash_forbidden_credentials()`. Keys are var names, values the
# snapshotted string. A name absent here means "not stashed" — `get_credential`
# falls through to `os.getenv`.
_STASH: dict[str, str] = {}


def stash_forbidden_credentials() -> list[str]:
    """Snapshot every present `FORBIDDEN_ENV_VARS` value, then delete it from
    `os.environ`. Returns the names moved (for logging). Idempotent.

    Call once at startup, right after config load, when the Claude Code harness
    is enabled. After this, the `claude` subprocess cannot inherit these vars.
    """
    moved: list[str] = []
    for name in FORBIDDEN_ENV_VARS:
        val = os.environ.get(name)
        if val in (None, ""):
            continue
        _STASH[name] = val
        del os.environ[name]
        moved.append(name)
    if moved:
        logger.info(
            "credential stash: moved %d var(s) out of os.environ for harness "
            "coexistence: %s",
            len(moved),
            ", ".join(moved),
        )
    return moved


def get_credential(name: str) -> str | None:
    """Read a credential: stash first (if scrubbed), else live `os.environ`.

    Model builders use this instead of `os.getenv` so they work whether or not
    the scrub ran.
    """
    if name in _STASH:
        return _STASH[name]
    return os.getenv(name)
