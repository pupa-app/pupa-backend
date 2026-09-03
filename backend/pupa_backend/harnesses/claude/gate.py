"""Tool-permission policy for the Claude Code agent loop.

The deterministic mechanism is a **PreToolUse hook** (`make_pre_tool_use_hook`):
the headless CLI in `permission_mode="default"` does **not** call `can_use_tool`
for tools it would otherwise auto-allow, but the PreToolUse hook fires before
*every* tool use. `can_use_tool` is kept as a non-asking backstop in case a CLI
version routes through it.

Policy (both paths share `_resolve`):
- **Frontend MCP tools** (`mcp__pupa_frontend__*`): allowed — they are the point
  of the loop. Honour the per-turn mute list `state["disabled_tools"]` (same key
  the LangGraph `ToolGatingMiddleware` reads) so iOS can hide a tool for a round.
- **Native host tools** (Read/Grep/Glob/Edit/Bash/Write/...): gated by
  `PUPA_CLAUDE_LOOP_NATIVE` — ``off`` (default, cloud-pinned) blocks all; ``read``
  allows Read/Grep/Glob; ``edit`` additionally allows Edit/Write/Bash.
- **Write/command tools, when permitted, ask the user first**.
  Unless `PUPA_CLAUDE_LOOP_ASK_PERMISSION=0`, an edit-class native tool parks and
  surfaces an approval request to the Pupa user as plain chat text; the user's
  next message (yes/no) resolves it. Read-class tools are auto-allowed.
- **Everything else**: denied (fail-closed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Awaitable, Callable

from ag_ui.core import EventType
from ag_ui.core.events import (
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from . import events as cl_events
from .frontend_tools import TOOL_PREFIX, bare_name

logger = logging.getLogger("uvicorn.error")

_NATIVE_READ = frozenset({"Read", "Grep", "Glob", "NotebookRead", "TodoWrite"})
_NATIVE_EDIT = frozenset({"Edit", "Write", "Bash", "NotebookEdit", "MultiEdit"})
# Read-only network tools — auto-approved in `full` scope (no host mutation).
_NATIVE_WEB = frozenset({"WebFetch", "WebSearch"})
# Skill / slash-command dispatch tools — invoking a skill is harmless; any host
# tool the skill then uses still hits this hook and is gated on its own.
_NATIVE_META = frozenset({"Skill", "SlashCommand"})
# Tools that never need a user prompt: local reads + web reads + skill dispatch.
_AUTO_NATIVE = _NATIVE_READ | _NATIVE_WEB | _NATIVE_META

# Words in a user's reply that grant a parked permission request. Anything else
# (including ambiguous text) denies — fail-closed.
_APPROVE_WORDS = frozenset({
    "yes", "y", "ok", "okay", "sure", "approve", "approved", "allow", "allowed",
    "go ahead", "do it", "proceed", "confirm", "confirmed", "yep", "yeah",
})
# Replies that approve AND switch the thread to run-freely (no more prompts).
_ALWAYS_PHRASES = (
    "always", "auto", "yes to all", "approve all", "allow all", "don't ask",
    "dont ask", "stop asking", "run freely", "yolo",
)

CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResultAllow | PermissionResultDeny],
]


_DEFAULT_NATIVE_SCOPE = "full"  # permissive by default — choosing the loop = wanting Claude Code's power


def resolve_native_scope(state: dict[str, Any] | None = None) -> str:
    """`claude_loop_native`: off | read | edit | full (default **full**).

    - off  — no host tools (cloud-pinned).
    - read — local read tools (Read/Grep/Glob...); maps to Claude's *plan* mode.
    - edit — read + write/command tools (Edit/Write/Bash...).
    - full — the **entire** native Claude Code toolset (web, subagents, …).

    The app may switch the mode **per turn** via `state["claude_loop_native"]`
    (e.g. flip between plan=`read` and edit=`full`) without a restart.
    """
    raw = None
    if isinstance(state, dict):
        raw = state.get("claude_loop_native") or state.get("claudeLoopNative")
    if not raw:
        raw = os.getenv("PUPA_CLAUDE_LOOP_NATIVE")
    scope = (str(raw).strip().lower() if raw else _DEFAULT_NATIVE_SCOPE)
    if scope in ("all", "host"):  # friendly aliases for "full"
        return "full"
    if scope not in ("off", "read", "edit", "full"):
        return _DEFAULT_NATIVE_SCOPE
    return scope


def native_enabled(state: dict[str, Any] | None = None) -> bool:
    return resolve_native_scope(state) in ("read", "edit", "full")


def _native_allowed(tool_name: str, scope: str) -> bool:
    if scope == "read":
        return tool_name in _NATIVE_READ
    if scope == "edit":
        return tool_name in _NATIVE_READ or tool_name in _NATIVE_EDIT
    if scope == "full":
        return True  # any non-frontend native/built-in tool is permitted
    return False  # "off" or anything unrecognised → fail-closed


def auto_approved_native_tools(state: dict[str, Any] | None = None) -> list[str]:
    """Native tools to pre-approve via `ClaudeAgentOptions.allowed_tools`.

    Only **read-class** (and, in ``full``, read-only web) tools — listing a tool in
    `allowed_tools` makes the SDK auto-approve it without the gate, so mutating
    tools are deliberately left out: they route through the PreToolUse hook. Empty
    when the scope is ``off``.
    """
    scope = resolve_native_scope(state)
    if scope in ("read", "edit"):
        return sorted(_NATIVE_READ)
    if scope == "full":
        return sorted(_AUTO_NATIVE)
    return []


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _require_approval() -> bool:
    """Whether mutating/command tools must be approved per use.

    **Default is False** — flow over friction: the loop runs commands freely. Opt
    back into prompts with `claude_loop_require_approval: true`
    (`PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL=1`); the legacy
    `PUPA_CLAUDE_LOOP_ASK_PERMISSION=1` also forces it on.
    """
    if _truthy(os.getenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL")):
        return True
    return _truthy(os.getenv("PUPA_CLAUDE_LOOP_ASK_PERMISSION"))


def _global_auto_approve() -> bool:
    """`PUPA_CLAUDE_LOOP_AUTO_APPROVE` (config `claude_loop_auto_approve: true`).

    Run every permitted command freely with no prompt — flow over friction. A clean
    positive bool, so the YAML loader maps `true → "1"` without the false-omit gotcha.
    """
    return (os.getenv("PUPA_CLAUDE_LOOP_AUTO_APPROVE") or "").strip().lower() in ("1", "true", "yes", "on")


def _auto_approve_from_state(state: dict[str, Any] | None) -> bool:
    """Per-turn run-freely flag the Pupa app can send in `RunAgentInput.state`.

    Lets the app expose an "auto-approve commands" toggle without backend changes —
    set `state["claude_loop_auto_approve"] = true` (or `autoApprove`).
    """
    if not isinstance(state, dict):
        return False
    val = state.get("claude_loop_auto_approve")
    if val is None:
        val = state.get("autoApprove")
    return bool(val)


def interpret_approval(text: str | None) -> bool:
    """Map a user's free-text reply to allow (True) / deny (False). Ambiguous → deny."""
    if not text:
        return False
    if interpret_always(text):
        return True
    t = text.strip().lower()
    if t in _APPROVE_WORDS:
        return True
    # Allow a leading approval word ("yes, go ahead", "ok do it").
    first = t.split(",")[0].split(".")[0].strip()
    if first in _APPROVE_WORDS:
        return True
    return any(phrase in t for phrase in ("go ahead", "do it", "approve", "allow it", "permission granted"))


def interpret_always(text: str | None) -> bool:
    """True if the reply approves AND asks to stop prompting for the rest of the thread."""
    if not text:
        return False
    t = text.strip().lower()
    return any(p in t for p in _ALWAYS_PHRASES)


def _describe(tool_name: str, tool_input: dict[str, Any]) -> str:
    """One-line human description of a native tool call for the approval prompt."""
    if tool_name == "Bash":
        cmd = tool_input.get("command") if isinstance(tool_input, dict) else None
        return f"run a shell command: `{cmd}`" if cmd else "run a shell command"
    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = tool_input.get("file_path") or tool_input.get("path") if isinstance(tool_input, dict) else None
        return f"modify `{path}`" if path else "modify a file"
    return f"use the `{tool_name}` tool"


def _emit_prompt(session: Any, text: str) -> bool:
    """Push an assistant-text message (start/content/end) to the session.

    Routed, not emitted: with no run open (the loop asking mid-way through a
    background task's injected turn) the prompt is held for the user's next run
    instead of landing on the queue ahead of that run's `RunStarted`.

    Returns whether it went out on a live run — i.e. whether the user can be
    expected to have seen it. See `_ask_user`.
    """
    mid = f"perm_{uuid.uuid4().hex}"
    delivered = session.route(
        TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant")
    )
    session.route(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=text))
    session.route(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid))
    return bool(delivered)


def _disabled_set(state: dict[str, Any] | None) -> set[str]:
    if isinstance(state, dict):
        raw = state.get("disabled_tools") or []
        if isinstance(raw, (list, tuple, set)):
            return {str(t) for t in raw}
    return set()


def _resolve_static(tool_name: str, disabled: set[str], scope: str = _DEFAULT_NATIVE_SCOPE) -> tuple[bool, bool, str]:
    """Decide a tool without the user-ask path, given the resolved native `scope`.

    Returns ``(allow, needs_ask, reason)``. `needs_ask` is True only for a permitted
    mutating/command native tool (the caller may then prompt the user).
    """
    if tool_name.startswith(TOOL_PREFIX):
        bare = bare_name(tool_name)
        if bare in disabled:
            return False, False, f"tool {bare!r} is muted for this turn"
        return True, False, ""
    if tool_name.startswith("mcp__"):
        # Operator-configured external MCP server tool — allowed without a prompt
        # (the operator opted in by configuring the server in config.yml).
        return True, False, ""
    if not _native_allowed(tool_name, scope):
        return False, False, (
            f"tool {tool_name!r} is not permitted in the Claude Code agent loop "
            f"(native scope={scope!r})"
        )
    if tool_name in _AUTO_NATIVE:
        return True, False, ""
    # Permitted mutating/command tool (edit-class, or anything else in `full`).
    return True, True, ""


def make_can_use_tool(state: dict[str, Any] | None, session: Any = None) -> CanUseTool:
    """Backstop SDK permission callback (the PreToolUse hook is the real gate).

    Does **not** prompt the user — it only enforces scope/mute so that, if some CLI
    version routes through `can_use_tool` instead of the hook, nothing forbidden
    slips through. Edit-class tools are allowed here; the prompt happens in the hook.
    """
    disabled = _disabled_set(state)
    scope = resolve_native_scope(state)

    async def _can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        allow, _needs_ask, reason = _resolve_static(tool_name, disabled, scope)
        if allow:
            return PermissionResultAllow()
        return PermissionResultDeny(message=reason, interrupt=False)

    return _can_use_tool


def make_pre_tool_use_hook(state: dict[str, Any] | None, session: Any = None):
    """Return a PreToolUse hook callback that gates every tool use.

    Signature is the SDK's `HookCallback`: ``async (input_data, tool_use_id, ctx)``.
    Returns a PreToolUse `permissionDecision` of ``allow`` / ``deny``. For a
    permitted edit-class native tool it parks and asks the Pupa user first (unless
    `PUPA_CLAUDE_LOOP_ASK_PERMISSION=0` or there's no session).
    """
    disabled = _disabled_set(state)
    state_auto = _auto_approve_from_state(state)
    scope = resolve_native_scope(state)

    async def _hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "") if isinstance(input_data, dict) else ""
        tool_input = (input_data.get("tool_input") or {}) if isinstance(input_data, dict) else {}
        allow, needs_ask, reason = _resolve_static(tool_name, disabled, scope)

        # Run freely (default) unless approval is explicitly required AND no
        # auto-approve override (global flag, per-turn state, or "always" this
        # thread) is set. Otherwise ask the user for this mutating/command tool.
        run_freely = (
            not _require_approval()
            or _global_auto_approve()
            or state_auto
            or (session is not None and session.auto_approve)
        )
        if allow and needs_ask and session is not None and not run_freely:
            allow = await _ask_user(session, tool_name, tool_input)
            reason = "" if allow else "the user denied permission to run this tool"

        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow" if allow else "deny",
                "permissionDecisionReason": reason or ("allowed" if allow else "denied"),
            }
        }

    return _hook


async def _ask_user(session: Any, tool_name: str, tool_input: dict[str, Any]) -> bool:
    """Park the model and surface an approval request to the Pupa user.

    Emits the request as plain assistant text, ends the current SSE turn, and waits
    on a per-session future the endpoint resolves from the user's next message.
    Returns True if the user approved.

    **Whether the question was delivered is recorded, not inferred.** The endpoint
    treats the user's next message as the yes/no, so a question that never left
    the backlog — the loop asking during a turn nobody requested — would silently
    consume a message the user meant as a new request. Neither `run_open` nor the
    background hold identifies that state on its own (a turn parked on a frontend
    tool call has no open run either, and its ask *is* delivered on the resume's
    SSE). `pending_decision_delivered` is the fact itself: the endpoint denies an
    undelivered ask rather than answering it from a message the user wrote blind.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    session.pending_decision = fut

    session.pending_decision_delivered = _emit_prompt(
        session,
        f"I need your permission to {_describe(tool_name, tool_input)}. "
        "Reply **yes** to allow, **no** to deny, or **always** to run everything "
        "this session without asking again.",
    )
    # Only a run that is open has an SSE to end.
    if session.run_open:
        session.emit(cl_events.run_finished(session.thread_id, session.current_run_id or ""))
        session.mark_interrupt()

    return bool(await fut)
