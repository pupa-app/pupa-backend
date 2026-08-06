"""Subscription-only, fail-closed billing controls for the Claude Code agent loop.

This module is load-bearing. The loop drives the `claude` CLI (the Agent SDK
spawns the binary), so it inherits Claude Code's auth/billing resolution. We must
bill **only** against the Claude Code subscription (Pro/Max OAuth — interactive
host login or a `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`). If
subscription billing can't be guaranteed, we **refuse to run**; we never silently
fall back to per-token API credits.

## Why detect-and-refuse, not silent-scrub

Confirmed against `claude-agent-sdk==0.2.106`: the SDK subprocess transport builds
its child env as ``{**os.environ (minus CLAUDECODE), **options.env}``. `options.env`
is an *overlay* — it can set/override vars but **cannot delete** a var already in
the parent `os.environ`. There is no "don't inherit parent env" switch.

Consequence: a strict allowlist passed via `options.env` does **not** strip an
inherited `ANTHROPIC_API_KEY`. And in Claude Code's auth precedence
`ANTHROPIC_API_KEY` outranks the subscription token — so a stray key silently
incurs API cost. Therefore enforcement is **detection + refuse-to-start**, not a
quiet scrub: if any forbidden credential var is present in the parent env we raise
`SubscriptionBillingUnavailable` (this is exactly the negative-test behaviour).

## Controls (per the design handoff §3)

1. Forbidden-var assertion (allowlist intent, fail-closed): the parent env must
   carry none of the API/Bedrock/Vertex credential vars. Raise if any present.
2. Isolated `CLAUDE_CONFIG_DIR` (optional, opt-in via `PUPA_CLAUDE_LOOP_CONFIG_DIR`)
   so a user-settings `apiKeyHelper` can't reintroduce a key. NOTE: by default we
   do NOT isolate, because the host's interactive subscription login lives in the
   default config dir / Keychain — isolating it would break `authMethod=claudeai`.
   Control 3 below is the real guard: the auth-status probe runs with the *same*
   config dir, so an `apiKeyHelper`-injected key surfaces as `authMethod=api_key`
   and is refused anyway.
3. Pre-flight auth-status probe: run `claude auth status --json` with the built
   env and assert `loggedIn=true`, `apiProvider=firstParty`, and
   `authMethod ∈ {claudeai, oauth_token}`. Anything else (api_key, third_party,
   none, or unrecognised) → refuse. Ambiguous == failure.
4. No silent alternate-billing mode. Subscription is the only supported path for
   v1. An api-billing path would have to be double-gated and is intentionally not
   implemented here.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

# Reuse the project's system prompt — the loop drives the same frontend tools.
from pupa_backend.prompts import SYSTEM_PROMPT

# Appended for the loop only. Claude Code's interactive affordances (popup
# questions, plan-mode approval, permission dialogs) have no surface in the Pupa
# chat UI — if the model used them the user would just see the turn stop with no
# visible reason. So tell the model to ask in plain chat text and stop, which the
# Pupa user answers with their next message (the loop resumes the same session).
_ASK_SUFFIX = (
    "\n\n### Asking the user\n"
    "You are running inside the Pupa chat app. You have NO way to pop up dialogs, "
    "questionnaires, or approval prompts. When you need a decision, a clarification, "
    "or any input from the user, simply ask them directly in your reply in plain "
    "text and stop — their next message is the answer. Never wait silently."
)

# Some frontend capabilities are gated behind a `get_tools_<group>` activation
# tool; the group's real tools attach only after the activation runs. The backend
# can't hot-add tools to a turn in progress, so it ends the current turn and
# automatically continues with the group attached. Tell the model to STOP after
# activating rather than flailing to find a tool that isn't attached yet — else it
# wastes a turn calling ToolSearch/Bash hunting for it (observed behaviour).
_ACTIVATION_SUFFIX = (
    "\n\n### Activating tool groups\n"
    "Some tools are gated behind a `get_tools_*` activation tool (e.g. "
    "`get_tools_tracker`). Calling one unlocks a group, but the group's tools are "
    "NOT attached to the current turn. So: call the activation tool, then end your "
    "turn without trying to use the unlocked tools yet. They will be attached "
    "automatically and you'll immediately be asked to continue — use them then. "
    "Do not search for or guess at a tool that isn't in your current tool list."
)

# Added when native host tools are enabled (`PUPA_CLAUDE_LOOP_NATIVE != off`). The
# base prompt frames the agent as Pupa-canvas-only ("you only see and act on what
# FRONTEND TOOLS allow"); without this the model disclaims shell/file access even
# when it has it. Mutating tools still trigger a per-use permission prompt.
_HOST_TOOLS_SUFFIX = (
    "\n\n### Host machine tools\n"
    "In addition to the Pupa frontend tools, you ARE running on the user's host "
    "machine with direct access to Claude Code's own tools — reading and editing "
    "files, running shell commands, searching the web, and so on. You can use them "
    "to do real work on the host (investigate code, run builds, edit files). When a "
    "tool would modify files or run a command, the user is asked to approve it "
    "first, so act normally and let the approval prompt handle safety. Do not claim "
    "you lack shell or file access — you have it."
)


def loop_skills() -> list[str] | str | None:
    """`PUPA_CLAUDE_LOOP_SKILLS`: ``all`` | comma-list of names | off (default).

    Returns the `ClaudeAgentOptions.skills` value (``"all"``, a list, or None).
    """
    raw = (os.getenv("PUPA_CLAUDE_LOOP_SKILLS") or "").strip()
    if not raw or raw.lower() in ("off", "0", "false", "no"):
        return None
    if raw.lower() == "all":
        return "all"
    return [s.strip() for s in raw.split(",") if s.strip()]


def loop_setting_sources() -> list[str]:
    """`setting_sources` for the loop.

    Skills are discovered from the user/project settings, so when skills are
    enabled we must load those sources. Otherwise we isolate (empty list) so the
    host's `settings.json` can't pre-approve tools (bypassing the permission gate)
    or inject an `apiKeyHelper` (billing). Enabling skills re-opens that surface —
    a deliberate, operator-controlled trade-off on a self-hosted install.
    """
    return ["user", "project"] if loop_skills() is not None else []


def loop_system_prompt(state: dict | None = None) -> str:
    """System prompt for the loop, composed at request time.

    Includes the host-tools paragraph only when native tools are enabled (for the
    resolved per-turn scope) so the model doesn't disclaim host access it has.
    """
    from .gate import native_enabled

    prompt = SYSTEM_PROMPT + _ASK_SUFFIX + _ACTIVATION_SUFFIX
    if native_enabled(state):
        prompt += _HOST_TOOLS_SUFFIX
    return prompt

logger = logging.getLogger("uvicorn.error")


class SubscriptionBillingUnavailable(RuntimeError):
    """Raised at startup/registration when subscription billing can't be guaranteed."""


# Credential vars that would route billing away from the subscription. Their mere
# presence in the parent env is disqualifying — the SDK would inherit them and the
# CLI's precedence puts `ANTHROPIC_API_KEY` ahead of the subscription token.
FORBIDDEN_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_PROFILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS",
)

# Vars we explicitly pass through to the SDK subprocess. USER/LOGNAME are needed
# for the macOS Keychain to resolve the login keychain where an interactive
# subscription OAuth login is stored (same comment as claude_code_tool._build_env).
_ALLOWLIST_ENV_VARS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "CLAUDE_CODE_OAUTH_TOKEN",
)

# `authMethod` values from `claude auth status --json` that mean "subscription".
#   claude.ai    — interactive Pro/Max OAuth login (host Keychain / config dir).
#                  This is the live value the CLI reports (v2.1.x); "claudeai" is
#                  kept as a defensive alias in case a build emits the dotless form.
#   oauth_token  — long-lived token from `claude setup-token` (subscription-gated)
# Everything else (api_key, third_party=Bedrock/Vertex, none, unknown) is refused.
_SUBSCRIPTION_AUTH_METHODS: frozenset[str] = frozenset({"claude.ai", "claudeai", "oauth_token"})


def controlled_config_dir() -> str | None:
    """Return the `CLAUDE_CONFIG_DIR` to pin, or None to use the host default.

    Default is None so the host's interactive subscription login (stored in the
    default config dir / Keychain) keeps working. Set `PUPA_CLAUDE_LOOP_CONFIG_DIR`
    to isolate — only sensible when authing via `CLAUDE_CODE_OAUTH_TOKEN`, which is
    env-based and so survives config-dir isolation.
    """
    override = os.getenv("PUPA_CLAUDE_LOOP_CONFIG_DIR")
    if override and override.strip():
        Path(override).mkdir(parents=True, exist_ok=True)
        return override
    return None


def build_sdk_env() -> dict[str, str]:
    """Build the overlay env for the SDK subprocess (`ClaudeAgentOptions.env`).

    Allowlist-only: PATH/HOME/USER/LOGNAME/LANG/LC_ALL + CLAUDE_CODE_OAUTH_TOKEN,
    plus an isolated CLAUDE_CONFIG_DIR when opted in. We deliberately add **no**
    credential var. Because the SDK overlays this on top of the inherited
    `os.environ` (it can't delete inherited vars), the absence of forbidden vars in
    the parent env is enforced separately by `assert_no_forbidden_env()`.
    """
    env: dict[str, str] = {}
    for key in _ALLOWLIST_ENV_VARS:
        val = os.getenv(key)
        if val is not None:
            env[key] = val
    config_dir = controlled_config_dir()
    if config_dir is not None:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    return env


def assert_no_forbidden_env() -> None:
    """Raise if the parent env carries any billing-diverting credential var.

    This is the allowlist intent made fail-closed: since the SDK inherits the
    parent env and can't strip it, a present forbidden var means we cannot
    guarantee subscription billing → refuse.
    """
    present = [k for k in FORBIDDEN_ENV_VARS if os.getenv(k) not in (None, "")]
    if present:
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: the environment carries "
            f"credential var(s) that would divert billing off the subscription: "
            f"{', '.join(present)}. The loop is subscription-only — unset these and "
            "authenticate with a Claude Pro/Max login (`claude auth login`) or a "
            "`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`."
        )


def probe_auth_status(env: dict[str, str] | None = None) -> dict[str, object]:
    """Run `claude auth status --json` with the built env; return the parsed dict.

    Raises `SubscriptionBillingUnavailable` if the CLI is missing, errors, or emits
    unparseable output (ambiguous == failure).
    """
    binary = os.getenv("CLAUDE_CODE_BIN") or shutil.which("claude") or "claude"
    probe_env = dict(env if env is not None else build_sdk_env())
    try:
        proc = subprocess.run(
            [binary, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=probe_env,
        )
    except FileNotFoundError as exc:
        raise SubscriptionBillingUnavailable(
            f"Refusing to start the Claude Code agent loop: `{binary}` not found on "
            "PATH. Install the Claude Code CLI or set CLAUDE_CODE_BIN."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: `claude auth status` "
            "timed out."
        ) from exc

    raw = (proc.stdout or "").strip()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: could not parse "
            f"`claude auth status --json` output (exit {proc.returncode}). "
            "Treating ambiguous auth state as failure."
        ) from exc
    if not isinstance(data, dict):
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: unexpected "
            "`claude auth status --json` shape (ambiguous == failure)."
        )
    return data


def assert_subscription_billing() -> dict[str, object]:
    """Fail-closed pre-flight: subscription billing must be guaranteed or we refuse.

    Combines the three controls and returns the parsed auth-status dict on success
    (handy for logging the resolved auth method). Raises
    `SubscriptionBillingUnavailable` on any failure.
    """
    # Control 4: the api-billing path is not implemented; loudly refuse if asked.
    billing = (os.getenv("PUPA_CLAUDE_LOOP_BILLING") or "subscription").strip().lower()
    if billing != "subscription":
        raise SubscriptionBillingUnavailable(
            f"PUPA_CLAUDE_LOOP_BILLING={billing!r} is not supported. The Claude Code "
            "agent loop is subscription-only in this build; remove the override."
        )

    # Control 1: no forbidden credential vars in the parent env.
    assert_no_forbidden_env()

    # Control 3: probe the resolved auth source and require subscription/OAuth.
    env = build_sdk_env()
    data = probe_auth_status(env)
    logged_in = bool(data.get("loggedIn"))
    auth_method = str(data.get("authMethod") or "")
    api_provider = str(data.get("apiProvider") or "")

    allowed = _allowed_auth_methods()
    if not logged_in:
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: not logged in. Run "
            "`claude auth login` (Pro/Max) or set CLAUDE_CODE_OAUTH_TOKEN from "
            "`claude setup-token`."
        )
    if api_provider != "firstParty":
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: auth resolves to "
            f"apiProvider={api_provider!r} (e.g. Bedrock/Vertex), not the "
            "first-party subscription. Subscription billing only."
        )
    if auth_method not in allowed:
        raise SubscriptionBillingUnavailable(
            "Refusing to start the Claude Code agent loop: auth method "
            f"{auth_method!r} is not a subscription credential (expected one of "
            f"{sorted(allowed)}). Refusing rather than billing per-token API credits."
        )

    logger.info(
        "claude_code loop: subscription billing confirmed (authMethod=%s, "
        "apiProvider=%s).",
        auth_method,
        api_provider,
    )
    return data


def _allowed_auth_methods() -> frozenset[str]:
    """Subscription auth methods, optionally extended via env for forward-compat.

    `PUPA_CLAUDE_LOOP_ALLOWED_AUTH_METHODS` (comma-separated) can add methods if a
    future CLI reports a new subscription string. Defaults stay fail-closed.
    """
    extra = os.getenv("PUPA_CLAUDE_LOOP_ALLOWED_AUTH_METHODS")
    if not extra:
        return _SUBSCRIPTION_AUTH_METHODS
    extras = {m.strip() for m in extra.split(",") if m.strip()}
    return _SUBSCRIPTION_AUTH_METHODS | extras
