"""AG-UI run handler for the Claude Code agent harness.

Mounted at `POST /harnesses/claude_code` (and `/` when it is the default harness)
by `harnesses.ClaudeCodeHarness` (see `app.py`). Owns the AG-UI interrupt/resume
round-trip for frontend tools by keeping a `ClaudeSDKClient` parked in the
per-thread `registry` across the two HTTP requests.

Lifecycle of one user turn:
  1. **New-turn POST** (no `forwardedProps.command.resume`): create a `LiveSession`,
     build the in-process frontend MCP server from `input.tools`, construct the
     `ClaudeSDKClient`, `query()` the latest user message, and start the **pump**
     task draining `receive_response()` into AG-UI events. Stream those events back.
  2. When Claude calls frontend tools the pump emits one batched `on_interrupt` +
     `RunFinished`, ending this SSE while the client stays parked.
  3. **Resume POST** (`command.resume.tool_results`): resolve the parked futures
     with the on-device results and re-attach a fresh SSE to the same queue; the
     pump continues until the next interrupt or the final `ResultMessage`.

Subscription-only billing is asserted at registration time (fail-closed) — see
`env.assert_subscription_billing`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    StreamEvent,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
)
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from pupa_backend.agui.tool_results import parse_tool_results
from pupa_backend.sse_replay import register_reattach_observer

from . import events, prompt_dump, registry, usage
from .config_mcp import SERVER_NAME as CONFIG_MCP_SERVER
from .config_mcp import build_config_mcp, config_tool_specs
from .env import (
    assert_subscription_billing,
    build_sdk_env,
    loop_setting_sources,
    loop_skills,
    loop_system_prompt,
)
from .frontend_tools import (
    SERVER_NAME,
    build_frontend_mcp,
    frontend_qualified_names,
    frontend_tool_specs,
)
from .models import LOOP_MODEL_ALIASES, is_loop_model
from .thinking import resolve_thinking
from .gate import (
    auto_approved_native_tools,
    interpret_approval,
    interpret_always,
    make_can_use_tool,
    make_pre_tool_use_hook,
    resolve_native_scope,
)

logger = logging.getLogger("uvicorn.error")

# `ResultMessage.origin["kind"]` values that mean "the session ran this turn on
# its own". See `_is_injected_turn` for why this is an allow-list.
_INJECTED_TURN_KINDS = frozenset({"task-notification"})

# Task lifecycle messages the CLI emits for subagents and background shell jobs.
# Folded into the session's background tracker; see `_track_task`.
_TASK_MESSAGES = (
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
    TaskUpdatedMessage,
)


def _default_loop_model() -> str:
    """The loop's fallback model: `CLAUDE_CODE_MODEL` else Opus 4.8."""
    return os.getenv("CLAUDE_CODE_MODEL") or "claude-opus-4-8"


def _resolve_loop_model(input: RunAgentInput) -> str:
    """Pick the model for this turn: per-request `forwardedProps.llm.model`
    (wins), else `CLAUDE_CODE_MODEL`, else Opus 4.8.

    The loop reads `input.forwarded_props` raw (no `ag_ui_langgraph` snake_case
    normalisation — same as the `command`/`resume` reads below), so the lowercase
    `llm`/`model` keys land here as-is. The `provider` field is ignored: the loop
    is Claude-only, so e.g. `anthropic/sonnet` and `bedrock/sonnet` both resolve
    to the alias `sonnet`.

    A non-Claude pick (e.g. an OpenRouter slug carried in from an imported
    `.pupa` bundle whose origin backend used a different harness) does NOT fail
    the turn: it would be hostile to break every conversation in an imported app
    just because it remembers a model this backend can't run. Instead we log a
    warning and fall back to the default Claude model. The user can switch the
    per-app model from the picker to make the choice explicit.
    """
    fp = input.forwarded_props or {}
    llm = fp.get("llm") if isinstance(fp, dict) else None
    requested = (llm.get("model") if isinstance(llm, dict) else None) or None
    if requested:
        if not is_loop_model(requested):
            fallback = _default_loop_model()
            logger.warning(
                "claude_code loop: requested model %r can't run on the Claude "
                "Code subscription loop (likely from an imported bundle) — "
                "falling back to %r. Pick a Claude alias %s to silence this.",
                requested,
                fallback,
                [a for a, _ in LOOP_MODEL_ALIASES],
            )
            return fallback
        return requested
    return _default_loop_model()


def _advertised_tools(input: RunAgentInput, resume_payload: Any) -> list[Any]:
    """Frontend tool descriptors advertised on a resume POST.

    iOS recomputes its tool surface right after running the dispatched tools and
    puts it in `command.resume.tools_after_round` (the descriptor list); it also
    re-sends the same set as `input.tools` on the resume round. Prefer the former
    (it's the post-dispatch snapshot) and fall back to `input.tools`. Each entry
    is a `{name, description, parameters}` dict `frontend_tools` understands.
    """
    if isinstance(resume_payload, dict):
        after = resume_payload.get("tools_after_round")
        if isinstance(after, list) and after:
            return after
    return list(input.tools or [])


def _latest_user_text(messages: list[Any]) -> str:
    """Newest user message rendered to text (content may be str or content parts)."""
    for msg in reversed(messages or []):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return _coerce_content(content)
    return ""


def _render_transcript(messages: list[Any]) -> str:
    """Flatten the whole conversation into a prompt — used only on the first turn
    of a thread (no SDK session to `resume` from yet)."""
    lines: list[str] = []
    for msg in messages or []:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = _coerce_content(content)
        if text:
            lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif getattr(p, "text", None):
                parts.append(p.text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _msg_content(msg: Any) -> Any:
    if msg is None:
        return None
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    return content


def _latest_user_message(messages: list[Any]) -> Any:
    for msg in reversed(messages or []):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "user":
            return msg
    return None


def _image_block(part: Any) -> dict[str, Any] | None:
    """AG-UI image part -> Anthropic image content block, or None if unusable.

    Handles both `InputContentSource` shapes — inline base64 (`data`) and URL
    references — and accepts either the pydantic `ImageInputContent` object or a
    plain dict.
    """
    source = getattr(part, "source", None)
    if source is None and isinstance(part, dict):
        source = part.get("source")
    if source is None:
        return None

    def _field(name: str) -> Any:
        val = getattr(source, name, None)
        if val is None and isinstance(source, dict):
            val = source.get(name)
        return val

    value = _field("value")
    if not value:
        return None
    if _field("type") == "url":
        return {"type": "image", "source": {"type": "url", "url": value}}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _field("mime_type") or "image/jpeg",
            "data": value,
        },
    }


def _image_blocks(content: Any) -> list[dict[str, Any]]:
    """Anthropic image blocks for every image part in a message's content.

    Without this, images attached to a user message are silently dropped —
    `_coerce_content` only keeps text — and never reach the model (issue: Claude
    loop does not receive photos/images from user messages).
    """
    if not isinstance(content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for p in content:
        ptype = getattr(p, "type", None) or (p.get("type") if isinstance(p, dict) else None)
        if ptype == "image":
            block = _image_block(p)
            if block is not None:
                blocks.append(block)
    return blocks


def _canonical_json(text: str) -> str:
    """Re-serialise a JSON payload with sorted keys; pass anything else through.

    The client builds these payloads from Swift `Dictionary`s, whose iteration
    order is randomised — so the same canvas/skills/type snapshot arrives with
    its keys shuffled on every turn. The bytes differ, the meaning doesn't, and
    since this block lands in the **system** prompt (which precedes `messages` in
    the cache prefix) each reshuffle re-cached the entire transcript behind it.

    Normalising here fixes it for every client, including builds already shipped
    that will never carry the client-side `.sortedKeys`. Array order is left
    alone — only object keys are sorted — and a value that isn't a JSON object or
    array is returned untouched.
    """
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(parsed, (dict, list)):
        return text
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _context_pairs(context: list[Any] | None) -> list[tuple[str, str]]:
    """AG-UI `input.context` as ordered `(description, value)` string pairs.

    Tolerates both the pydantic `Context` model and a plain dict. Shared by
    `_render_context` and the cache fingerprint, so the diagnosis hashes exactly
    the text that reaches the prompt. Values are canonicalised (see
    `_canonical_json`) so a client's key ordering can't bust the prompt cache.
    """
    pairs: list[tuple[str, str]] = []
    for entry in context or []:
        desc = getattr(entry, "description", None)
        if desc is None and isinstance(entry, dict):
            desc = entry.get("description")
        val = getattr(entry, "value", None)
        if val is None and isinstance(entry, dict):
            val = entry.get("value")
        pairs.append(((desc or "").strip(), _canonical_json((val or "").strip())))
    return pairs


def _render_context(context: list[Any] | None) -> str:
    """Flatten AG-UI `input.context` into a text block (`description` then its
    stringified `value`, entries blank-line separated).

    The loop builds its own prompt, so — unlike the `ag_ui_langgraph` harness,
    which dumps context into the system prompt — the ambient context the
    frontend pushes every turn (live canvas state, memories snapshot, the MyApp
    system prompt / AGENTS.md) only reaches the model if rendered in here.
    """
    blocks: list[str] = []
    for desc, val in _context_pairs(context):
        block = f"{desc}\n{val}".strip()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


# Header for the ambient-context block appended to the system prompt. Placed at
# the *end* so the stable `loop_system_prompt` prefix stays prompt-cacheable; the
# block itself is volatile (canvas state) but re-composed fresh each turn.
_CONTEXT_HEADER = (
    "Ambient context — supplied by the app each turn (live canvas state, memories, "
    "and MyApp instructions / AGENTS.md). Treat it as authoritative for this turn:"
)


def _compose_system_prompt(base: str, context: list[Any] | None) -> str:
    """Append the rendered ambient context to the base loop system prompt.

    The loop rebuilds options (hence the system prompt) on every new-turn and
    continuation POST, and the SDK passes `--system-prompt` on each subprocess
    spawn — including resumes — so the context refreshes *in place* each turn
    instead of accumulating in the message transcript.
    """
    block = _render_context(context)
    if not block:
        return base
    return f"{base}\n\n{_CONTEXT_HEADER}\n\n{block}"


def _query_content(text: str, image_blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Query payload for a turn: a plain string when text-only (unchanged
    behaviour), or a text+image block list when the turn carries images."""
    if not image_blocks:
        return text
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(image_blocks)
    return blocks


async def _send_query(client: ClaudeSDKClient, content: str | list[dict[str, Any]]) -> None:
    """Forward the turn's prompt to the SDK. Text-only turns keep the simple
    string path; multimodal turns stream a single structured user message so the
    image blocks reach the model (a plain string would drop them)."""
    if isinstance(content, str):
        await client.query(content or "")
        return

    async def _gen():
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
        }

    await client.query(_gen())


# CLI built-in/server tools the loop never wants (the gate denies them anyway, but
# disallowing up front saves a wasted model turn). `ToolSearch` leaks in despite a
# frontend-only `allowed_tools`; the interactive ones (`AskUserQuestion`,
# `ExitPlanMode`) have no Pupa surface — disallowing them makes the model ask in
# plain chat text instead (see `loop_system_prompt`), which the user can actually see.
_DISALLOWED_BUILTINS = ["ToolSearch", "AskUserQuestion", "ExitPlanMode"]


def _options_for(
    input: RunAgentInput,
    session: registry.LiveSession,
    mcp: Any,
    tools_descriptors: list[Any],
    resume_id: str | None,
) -> ClaudeAgentOptions:
    """Build `ClaudeAgentOptions` exposing `tools_descriptors` as the frontend MCP
    surface, resuming `resume_id`. Used for both the new-turn build (advertised
    `input.tools`, resume = remembered session id) and a mid-turn continuation
    (widened tools, resume = this turn's live session id). Records the resulting
    live frontend surface on the session so a later resume can detect a widening.
    """
    mcp_server, qualified = build_frontend_mcp(list(tools_descriptors or []), session)
    # The live client now exposes exactly `qualified`. A resume advertising more
    # than this is a gate unlock the frozen in-process server can't absorb — the
    # endpoint arms a continuation turn instead.
    session.frontend_qualified = set(qualified)
    # Config-driven MCP servers (config.yml `mcp_servers:`) are exposed via the
    # single shared connection bridged in-process — NOT handed to the claude
    # subprocess as `--mcp-config` (which it won't trust with `setting_sources=[]`).
    config_server, config_qualified = build_config_mcp(mcp)
    state = input.state
    scope = resolve_native_scope(state)
    # The gate's hooks close over `state`, so a later turn may only reuse this
    # client while everything they read is unchanged. See `_cannot_continue_live`.
    session.gate_state = _gate_state(state)
    # Frontend + config MCP tools + read-class native tools are pre-approved.
    # Mutating/command native tools are intentionally NOT here so the gate's
    # PreToolUse hook runs (listing a tool in allowed_tools bypasses the prompt).
    allowed = sorted(qualified | config_qualified) + auto_approved_native_tools(state)
    mcp_servers: dict[str, Any] = {SERVER_NAME: mcp_server}
    if config_server is not None:
        mcp_servers[CONFIG_MCP_SERVER] = config_server
    # `read` scope ≈ Claude's plan mode (investigate, don't change); otherwise the
    # gate is the authority so we stay in default permission mode.
    permission_mode = "plan" if scope == "read" else "default"
    base_system = loop_system_prompt(state)
    system = _compose_system_prompt(base_system, input.context)
    thinking = resolve_thinking(input) or {}
    model = _resolve_loop_model(input)
    # What this client is frozen with. A later turn may reuse it only if it wants
    # the same model and thinking level; the ambient context is re-delivered in
    # the query instead (see `_continue_live_turn`).
    session.client_shape = (model, tuple(sorted(thinking.items())))
    session.context_pairs = _context_pairs(input.context)
    skills = loop_skills()
    cwd = os.getenv("CLAUDE_CODE_WORKSPACE") or None
    # Anthropic caches on an exact `tools` → `system` → `messages` prefix and the
    # CLI owns the breakpoints, so anything drifting here re-writes the whole
    # cache instead of reading it. Log what moved since this thread's last turn.
    tool_specs = frontend_tool_specs(list(tools_descriptors or [])) + config_tool_specs(mcp)
    context_pairs = _context_pairs(input.context)
    fingerprint = usage.fingerprint(
        model=model,
        base_system=base_system,
        system=system,
        context_pairs=context_pairs,
        tool_specs=tool_specs,
        permission_mode=permission_mode,
        thinking=thinking,
        skills=skills,
        cwd=cwd,
    )
    logger.info(
        usage.cache_line(
            session.thread_id,
            fingerprint,
            tool_count=len(tool_specs),
            system_chars=len(system),
        )
    )
    # `PUPA_CLAUDE_PROMPT_DUMP=<dir>` (off by default — the payload is user data)
    # writes this exact prefix plus a unified diff against the thread's previous
    # turn, for when the fingerprint key isn't enough to see what moved.
    if prompt_dump.dump_dir() is not None:
        prompt_dump.write(
            session.thread_id,
            prompt_dump.build_payload(
                thread_id=session.thread_id,
                model=model,
                base_system=base_system,
                system=system,
                context_pairs=context_pairs,
                tool_specs=tool_specs,
                permission_mode=permission_mode,
                thinking=thinking,
                skills=skills,
                cwd=cwd,
                fingerprint=fingerprint,
            ),
        )
    return ClaudeAgentOptions(
        system_prompt=system,
        mcp_servers=mcp_servers,
        skills=skills,
        allowed_tools=allowed,
        disallowed_tools=_DISALLOWED_BUILTINS,
        can_use_tool=make_can_use_tool(state, session),
        # The real gate: a PreToolUse hook fires before every tool use (unlike
        # can_use_tool, which the headless CLI skips for auto-allowed tools). This
        # is where the per-command user-permission prompt happens.
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[make_pre_tool_use_hook(state, session)])]},
        env=build_sdk_env(),
        permission_mode=permission_mode,
        # Empty by default so host settings.json can't pre-approve tools or inject
        # an apiKeyHelper. When skills are enabled, user/project sources are loaded
        # (skills are discovered from them) — the PreToolUse hook still fires and is
        # the permission authority regardless of settings-level allow-rules.
        setting_sources=loop_setting_sources(),
        cwd=cwd,
        # Per-request model (iOS `forwardedProps.llm.model`) wins, else the
        # `CLAUDE_CODE_MODEL` config/env default, else Opus 4.8. See
        # `_resolve_loop_model`.
        model=model,
        # Per-request extended-thinking level (iOS `forwardedProps.llm.thinking`).
        # Spread as `thinking=` only when a known level was picked; otherwise the
        # option stays unset and the CLI/subscription default applies. See
        # `resolve_thinking`.
        **thinking,
        resume=resume_id,
        # Stream assistant text token-by-token: the SDK then yields partial
        # `StreamEvent`s that `_pump` maps to incremental `TextMessageContent`
        # deltas (parity with the deepagents harness). See `events.translate_stream_event`.
        include_partial_messages=True,
    )


# No explicit cap on continuations is needed: at most ONE runs per resume POST
# (`pending_widen_descriptors` is set only by the resume handler and cleared the
# moment a continuation consumes it, and a continuation can't self-trigger — its
# own ResultMessage sees it already None). A second continuation requires another
# iOS resume POST (a distinct activation), which iOS's own per-send round cap
# already bounds. So the chain is inherently bounded; don't re-add a counter.

# Synthetic user message that kicks off a continuation turn after a gate unlock.
# The turn resumes the SDK session (full prior context) with the widened tool
# set attached, so the model can finally use what it just activated.
_CONTINUATION_PROMPT = (
    "The tool group you just activated is now available. Continue with the user's "
    "request using the newly-available tools. Do not call a get_tools_* activation "
    "tool again for tools you already have."
)


async def _interrupt_for_widen(session: registry.LiveSession) -> bool:
    """Best-effort `interrupt()` of the parked client to stop a gate-unlock turn.

    Returns True if the interrupt was sent (the pump will then convert the
    resulting `ResultMessage` into a continuation), False if the client can't be
    interrupted — the caller drops the widen and lets the tools appear next turn.
    """
    client = session.client
    interrupt = getattr(client, "interrupt", None)
    if not callable(interrupt):
        return False
    try:
        await interrupt()
        return True
    except Exception:  # noqa: BLE001 — never fail the resume on an interrupt hiccup
        logger.exception("claude_code loop: interrupt for tool-unlock cut-over failed")
        return False


async def _start_continuation(session: registry.LiveSession, descriptors: list[Any]) -> None:
    """Swap the parked client for a fresh one exposing `descriptors` and resume.

    The old turn has already finished (its `ResultMessage` triggered this), so the
    SDK session transcript is well-formed and `session.sdk_session_id` is set. We
    build a new `ClaudeSDKClient` with the widened frontend surface + the same
    state/model, `resume` that session id, `query` a synthetic continue prompt, and
    hand the queue to a fresh pump. The new pump feeds the SAME session queue, so
    the still-attached `_stream` keeps delivering events onto the same SSE — the
    continuation is invisible to iOS beyond the extra tool events (no second
    `RunStarted`; the resume POST already emitted one for this run).
    """
    old_client = session.client
    resume_id = session.sdk_session_id
    # `_options_for` writes what the *new* client will be shaped like onto the
    # session. If the swap never happens, those fields describe a client the
    # session does not have — and `_cannot_continue_live` would then compare the
    # next turn against the wrong tool surface and never notice the drift.
    shape_before = (
        session.frontend_qualified, session.gate_state,
        session.client_shape, session.context_pairs,
    )

    def _restore_shape() -> None:
        (session.frontend_qualified, session.gate_state,
         session.client_shape, session.context_pairs) = shape_before

    options = _options_for(session.turn_input, session, session.turn_mcp, descriptors, resume_id)
    try:
        new_client = ClaudeSDKClient(options)
        await new_client.connect()
    except Exception:
        _restore_shape()
        raise
    logger.info(
        "claude_code loop: continuation turn (resume=%s, tools=%d) thread=%s",
        resume_id, len(session.frontend_qualified), session.thread_id,
    )
    # Swap only once the new child has actually taken the query. Swapping first
    # and failing here would leave `session.client` pointing at a child nobody
    # pumps while the old one stays connected — and, now that a session with
    # background work parks instead of finishing, that broken session would
    # survive and swallow the next turn.
    try:
        await new_client.query(_CONTINUATION_PROMPT)
    except Exception:
        _restore_shape()
        try:
            await new_client.disconnect()
        except Exception:  # noqa: BLE001 — best-effort teardown of the unused client
            logger.debug("claude_code loop: new client disconnect failed", exc_info=True)
        raise
    session.client = new_client
    if session.background.active:
        # The old child owned those tasks and is about to be disconnected: they
        # die with it. Counting them against the new child would park every
        # later turn on work nobody is doing, pinning the session indefinitely.
        logger.info(
            "claude_code loop: continuation replaces the child — orphaning %s "
            "(thread=%s)", session.background.summary(), session.thread_id,
        )
        session.background.reset()
    if old_client is not None:
        try:
            await old_client.disconnect()
        except Exception:  # noqa: BLE001 — best-effort teardown of the finished turn
            logger.debug("claude_code loop: old client disconnect failed", exc_info=True)
    import asyncio

    session.pump_task = asyncio.ensure_future(_pump(session))


def _gate_state(state: Any) -> tuple:
    """The parts of `RunAgentInput.state` the permission gate reads.

    A live client's `can_use_tool` / PreToolUse hook and its `allowed_tools` were
    built from one turn's state; a later turn that changes any of these needs a
    fresh client, not the parked one. Everything else in `state` is app data the
    gate never looks at, so it must not force a rebuild — that would orphan the
    session's background work on every turn.
    """
    if not isinstance(state, dict):
        return ()
    return (
        resolve_native_scope(state),
        bool(state.get("claude_loop_auto_approve") or state.get("autoApprove")),
        tuple(sorted(str(t) for t in (state.get("disabled_tools") or []))),
    )


def _is_injected_turn(msg: ResultMessage) -> bool:
    """True when this result answers a turn the CLI ran on its own.

    `ResultMessage.origin` is absent (or `{"kind": "human"}`) for a prompt this
    backend sent, and carries `{"kind": "task-notification"}` for a turn the
    session injected when a background task reported in. Those have no HTTP run
    waiting on them, so they must not emit a terminal AG-UI event.

    An **allow-list**, deliberately. The SDK's `kind` vocabulary is open and the
    dependency is floored rather than pinned, so "anything that is not human is
    injected" fails open: a new kind on a turn this backend prompted would emit no
    terminal at all and hang the user's request until the idle sweep. Misreading
    an injected turn as ours costs a terminal event on a run that has already
    ended — which is routed to the backlog, not the wire.
    """
    origin = getattr(msg, "origin", None)
    if not isinstance(origin, dict):
        return False
    kind = origin.get("kind")
    if kind in _INJECTED_TURN_KINDS:
        return True
    if kind not in (None, "human"):
        logger.info(
            "claude_code loop: treating unrecognised ResultMessage origin %r as this "
            "backend's own turn", kind,
        )
    return False


def _track_task(session: registry.LiveSession, msg: Any) -> None:
    """Fold one CLI task lifecycle message into the session's background tracker.

    `task_started` / `task_progress` mean the task is running. The two status
    messages are the only evidence a task *ended*, and a `task_updated` patch
    carrying only `end_time` / `result` parses to `status=None` — absence of a
    status is not evidence of anything, so it is dropped rather than read as a
    start (which would resurrect a finished task and pin the session open).
    """
    task_id = getattr(msg, "task_id", None)
    if isinstance(msg, (TaskStartedMessage, TaskProgressMessage)):
        session.background.start(task_id, getattr(msg, "description", "") or "")
        return
    status = getattr(msg, "status", None)
    if status is None and isinstance(getattr(msg, "patch", None), dict):
        status = msg.patch.get("status")
    if status is None:
        return
    if session.background.update(task_id, status):
        logger.info(
            "claude_code loop: background task %s -> %s (thread=%s, remaining=%s)",
            task_id, status, session.thread_id, session.background.summary(),
        )


def _cannot_continue_live(session: registry.LiveSession, input: RunAgentInput) -> str | None:
    """Why this turn can't run on the parked live session, or None if it can.

    Reusing the client is what keeps background work alive across turns, but it
    is only correct when the turn asks for nothing the live client was not built
    with: its in-process tool server is frozen at `connect()` and its permission
    hook closed over the previous turn's state.
    """
    if session.client is None:
        return "no client"
    pump = session.pump_task
    if pump is None or pump.done():
        return "pump not running"
    if session.has_unresolved_pending():
        return "a frontend tool call is still parked"
    if session.pending_decision is not None and not session.pending_decision.done():
        return "a permission request is still parked"
    advertised = frontend_qualified_names(list(input.tools or []))
    if advertised != session.frontend_qualified:
        # Not just widening: a turn that drops a tool would still run on a client
        # exposing it, and the model could then call something the app is not
        # offering this round — a call no resume will ever answer.
        return "the turn's frontend tool surface differs from the live client's"
    gate_state = _gate_state(input.state)
    if session.gate_state is not None and gate_state != session.gate_state:
        return f"permission state changed {session.gate_state} -> {gate_state}"
    shape = (_resolve_loop_model(input), tuple(sorted((resolve_thinking(input) or {}).items())))
    if session.client_shape is not None and shape != session.client_shape:
        return f"model/thinking changed {session.client_shape} -> {shape}"
    return None


async def _continue_live_turn(
    session: registry.LiveSession, input: RunAgentInput, run_id: str, mcp: Any
) -> StreamingResponse | None:
    """Run this turn on the already-connected client, keeping its background work.

    The SDK session already holds the whole conversation, so only the newest user
    message is sent — same as any resumed turn. The existing pump is still
    draining this client's stream and picks the turn up. Returns None if the
    query could not be sent, so the caller falls back to a fresh session.
    """
    session.turn_input = input
    session.turn_mcp = mcp
    latest_user = _latest_user_message(input.messages)
    text = _coerce_content(_msg_content(latest_user))
    image_blocks = _image_blocks(_msg_content(latest_user))
    # The ambient context normally refreshes in the system prompt on every fresh
    # client. This turn has no fresh client, so hand the model the new context in
    # the query — otherwise it answers on a snapshot from the previous turn.
    pairs = _context_pairs(input.context)
    if pairs != session.context_pairs:
        session.context_pairs = pairs
        block = _render_context(input.context)
        if block:
            text = f"{_CONTEXT_HEADER}\n\n{block}\n\n{text}"
    # Send before opening the run: a failure then costs nothing (the deferred
    # backlog is still intact for the fresh session to inherit), and anything the
    # pump produces in between is deferred and released by `open_run` in order.
    try:
        await _send_query(session.client, _query_content(text, image_blocks))
    except Exception:  # noqa: BLE001 — fall back to a fresh session
        logger.exception(
            "claude_code loop: continuing the live session failed; starting fresh"
        )
        return None
    session.open_run(run_id, events.run_started(input.thread_id, run_id))
    logger.info(
        "claude_code loop: continuing the live session (background: %s) thread=%s",
        session.background.summary(), session.thread_id,
    )
    return _stream(session)


async def _pump(session: registry.LiveSession) -> None:
    """Drain the SDK message stream into AG-UI events on the session queue.

    Spans the whole Claude turn — including frontend tool round-trips — and then
    keeps going: `receive_messages()` is the session's *whole* stream, so one pump
    serves every turn this live client runs. At each frontend batch it emits
    `on_interrupt` + `RunFinished` and parks (the iterator blocks until the resume
    POST resolves the tool futures). When a gate unlock armed
    `pending_widen_descriptors`, the endpoint interrupts this turn; the pump then
    converts the resulting `ResultMessage` into a continuation turn that exposes
    the widened tools instead of finishing (see `_start_continuation`).

    **Background work.** A turn that ends with background tasks still running
    parks (`mark_park`) instead of finishing: the run's SSE closes but the CLI
    child stays alive, so its tasks keep running and the injected turn they
    trigger on completion is received here. Those events are held in the
    session's deferred backlog and delivered on the next run. Without this the
    child is killed at the `ResultMessage` and every background subagent dies —
    see `pupa_backend.agui.background`.
    """
    client = session.client
    thread_id = session.thread_id
    # Per-turn text-streaming cursor for partial `StreamEvent`s (see below).
    stream_state = events.new_stream_state()

    out = session.route  # onto the open run's SSE, else into the backlog

    try:
        async for msg in client.receive_messages():
            # Remember the SDK session id from the EARLIEST message that carries it
            # (SystemMessage init / AssistantMessage), not just the final
            # ResultMessage — so a turn that errors or is interrupted before the
            # result still leaves a resumable id for the next turn. iOS re-sends
            # only user messages, so without this a mid-turn failure would drop all
            # assistant/tool context from the model's view (issue: context loss).
            sid = getattr(msg, "session_id", None)
            if sid and session.sdk_session_id != sid:
                session.sdk_session_id = sid
                registry.remember_session_id(thread_id, sid)

            if isinstance(msg, _TASK_MESSAGES):
                _track_task(session, msg)
                continue

            if isinstance(msg, StreamEvent):
                # Partial message: stream assistant text as incremental deltas. The
                # whole `AssistantMessage` still follows (handled below with
                # skip_text so the text isn't re-sent).
                for e in events.translate_stream_event(msg.event, stream_state):
                    out(e)
                continue

            if isinstance(msg, AssistantMessage):
                model = getattr(msg, "model", None)
                if model and session.sdk_model is None:
                    session.sdk_model = model
                    logger.info("claude_code loop: model=%s (thread=%s)", model, thread_id)
                # Per-API-call token + cache stats (yellow). The SDK yields one
                # `AssistantMessage` per content block, all carrying the same
                # message-level usage, so log once per message id — otherwise a
                # text+tool_use reply prints the identical line several times.
                if events.record_usage_logged(msg, stream_state):
                    token_line = usage.message_line(getattr(msg, "usage", None), thread_id)
                    if token_line:
                        logger.info(token_line)
                # Skip only the text that actually streamed above (avoids a duplicate
                # whole-block emit). Messages the CLI fabricates locally — rate-limit
                # notices, API errors, the "No response requested." reply to a query
                # queued behind a resumed session — never stream, so their text is
                # emitted here or the user sees a run with no content at all.
                evs, frontend_calls = events.translate_assistant_message(
                    msg, skip_text=events.text_already_streamed(msg, stream_state)
                )
                for e in evs:
                    out(e)
                if frontend_calls and session.pending_widen_descriptors is not None:
                    # A gate unlock is pending: the endpoint has already sent an
                    # interrupt to stop this (still-narrow) turn. Any tool call it
                    # squeezes out first is the model flailing for tools it can't
                    # reach yet — drop it (don't park iOS on it). The continuation
                    # turn, built at the imminent ResultMessage, has the real tools.
                    logger.info(
                        "claude_code loop: dropping %d pre-continuation tool call(s) %s",
                        len(frontend_calls), [c["name"] for c in frontend_calls],
                    )
                elif frontend_calls and not session.run_open:
                    # An injected turn (a background task reporting in) reaching
                    # for an on-device tool. No run is open, so no SSE can carry
                    # the interrupt and no resume will ever answer it — fail the
                    # calls now instead of blocking the loop on the park wall.
                    logger.info(
                        "claude_code loop: rejecting %d frontend call(s) %s from a "
                        "turn with no open run (thread=%s)",
                        len(frontend_calls), [c["name"] for c in frontend_calls], thread_id,
                    )
                    for c in frontend_calls:
                        await session.register_pending(
                            c["id"], c["name"], c["args"], run_id=None
                        )
                    await session.reject_pending(
                        [c["id"] for c in frontend_calls], "app_not_attached"
                    )
                elif frontend_calls:
                    for c in frontend_calls:
                        await session.register_pending(
                            c["id"], c["name"], c["args"], run_id=session.current_run_id
                        )
                    out(events.on_interrupt(frontend_calls))
                    out(events.run_finished(thread_id, session.current_run_id or ""))
                    session.mark_interrupt()
            elif isinstance(msg, ResultMessage):
                # Turn totals + cost (yellow). Logged before the continuation
                # hand-off so a widened continuation still reports the tokens
                # its predecessor turn burned.
                logger.info(usage.result_line(msg, thread_id))
                descriptors = session.pending_widen_descriptors
                if descriptors is not None and session.sdk_session_id:
                    # A gate tool unlocked more tools; the endpoint interrupted the
                    # narrow turn so we could re-attach the widened surface. Continue
                    # on a fresh resumed client that exposes it. This runs BEFORE the
                    # finish/error sentinel (and regardless of `is_error`, since the
                    # deliberate interrupt reports one), so the SSE stays open across
                    # the hand-off.
                    session.pending_widen_descriptors = None
                    try:
                        await _start_continuation(session, descriptors)
                        return  # the new pump now owns the queue
                    except Exception:  # noqa: BLE001 — never strand the turn on a failed continuation
                        logger.exception("claude_code loop: continuation failed; finishing turn")
                session.pending_widen_descriptors = None
                if _is_injected_turn(msg):
                    # A turn the session ran on its own (a background task
                    # reporting in). Nothing is waiting on it, so emit no
                    # terminal event — its output is already in the deferred
                    # backlog and rides the next run.
                    logger.info(
                        "claude_code loop: injected turn finished (origin=%s, "
                        "background=%s) thread=%s",
                        (msg.origin or {}).get("kind"), session.background.summary(), thread_id,
                    )
                    stream_state = events.new_stream_state()
                    continue
                if session.run_open:
                    out(
                        events.run_error(str(msg.result or "claude reported an error"))
                        if msg.is_error
                        else events.run_finished(thread_id, session.current_run_id or "")
                    )
                # A clean turn end with background work still running parks
                # instead of finishing: the run's SSE closes, the CLI child stays
                # alive, and the work reports through this same session. An
                # *errored* turn tears down as before — a wedged child must not
                # be kept on the strength of tasks it may never report.
                if not msg.is_error and session.background.active and session.background.hold():
                    logger.info(
                        "claude_code loop: parking for background work (%s) thread=%s",
                        session.background.summary(), thread_id,
                    )
                    session.mark_park()
                    stream_state = events.new_stream_state()
                    continue
                if msg.is_error:
                    session.mark_error()
                else:
                    session.mark_finish()
                return
            # SystemMessage / RateLimitEvent: nothing else to surface.
            # (StreamEvent is handled above.)
    except Exception as exc:  # noqa: BLE001 — surface any SDK failure to the client
        logger.exception("claude_code loop: pump failed")
        out(events.run_error(f"claude_code loop error: {exc}"))
        session.mark_error()


def _stream(session: registry.LiveSession) -> StreamingResponse:
    encoder = EventEncoder()

    async def _gen():
        async for event in registry.attach(session):
            yield encoder.encode(event)

    return StreamingResponse(_gen(), media_type="text/event-stream")


def _error_stream(message: str) -> StreamingResponse:
    encoder = EventEncoder()

    async def _gen():
        yield encoder.encode(events.run_error(message))

    return StreamingResponse(_gen(), media_type="text/event-stream")


def register_claude_loop_endpoint(
    app: FastAPI, path: str = "/", mcp: Any = None
) -> None:
    """Register the `POST {path}` Claude Code harness handler. Fail-closed on billing.

    `path` is the mount point — `/harnesses/claude_code` (and `/` for the default
    harness) so several harnesses can coexist in one server.

    `mcp` is the shared `MCPServersLifecycle` (config.yml `mcp_servers:`) opened
    once at startup; its tools are bridged in-process so every thread reuses the
    same connection (see `config_mcp.build_config_mcp`).
    """
    # Subscription pre-flight runs here so an unsafe deploy fails at startup.
    assert_subscription_billing()
    # A re-attach POST is short-circuited by the replay middleware and never
    # reaches this handler, so register a hook to hear about it — a returning
    # client must re-arm the liveness grace of a turn parked on its thread.
    # Registration is idempotent, so mounting on both `/` and `/harnesses/…` is fine.
    register_reattach_observer(registry.note_reattach)
    scope = resolve_native_scope(None)
    model = os.getenv("CLAUDE_CODE_MODEL") or "(Claude Code default)"
    logger.info(
        "claude_code agent harness active on POST %s — native scope=%s, model=%s",
        path, scope, model,
    )
    from .gate import _require_approval  # local import to avoid touching gate's public API

    if scope in ("edit", "full") and not _require_approval():
        logger.warning(
            "claude_code loop is PERMISSIVE: native host tools (scope=%s) run "
            "WITHOUT a per-command approval prompt. Anyone who can chat to this "
            "backend can run host commands. Keep it to your own machine; set "
            "claude_loop_require_approval: true to re-enable prompts, or "
            "claude_loop_native: \"off\"/\"read\" to restrict tools.",
            scope,
        )

    @app.post(path)
    async def claude_loop_endpoint(request: Request):  # noqa: ANN202 — FastAPI route
        body = await request.json()
        input = RunAgentInput.model_validate(body)
        thread_id = input.thread_id
        run_id = input.run_id

        forwarded = input.forwarded_props or {}
        command = forwarded.get("command") if isinstance(forwarded, dict) else None
        resume_payload = command.get("resume") if isinstance(command, dict) else None
        keepalive_payload = command.get("keepalive") if isinstance(command, dict) else None

        # --- Keepalive ping: client liveness while a frontend tool is parked ---
        # Fire-and-forget: touch the parked session's liveness clock and return
        # 204 — no run starts, no SSE. `state: "background"` suspends the grace
        # (iOS froze the client's timers); any other ping re-arms it. A missing
        # session is still a 204 no-op.
        if keepalive_payload is not None:
            session = registry.get(thread_id)
            if session is not None and not session.disposed:
                state = (
                    keepalive_payload.get("state")
                    if isinstance(keepalive_payload, dict) else None
                )
                await session.keepalive(backgrounded=(state == "background"))
            return Response(status_code=204)

        # --- Resume POST: deliver on-device tool results to the parked session ---
        if resume_payload is not None:
            session = registry.get(thread_id)
            # A session held open for background work is NOT parked on a tool
            # call: no handler is waiting, so attaching would hang the request on
            # a queue nothing will feed. Answer it like the stale-session case.
            if (
                session is not None
                and not session.disposed
                and session.background.holding
                and not session.has_unresolved_pending()
            ):
                logger.info(
                    "claude_code loop: resume POST for a thread with no parked tool "
                    "call (session held for background work) thread_id=%s", thread_id,
                )
                session = None
            if session is None or session.disposed:
                return _error_stream(
                    "no parked Claude Code session for this thread — the turn may have "
                    "timed out or the backend restarted. Start a new message."
                )
            session.open_run(run_id, events.run_started(thread_id, run_id))
            # Detect a gate unlock: this resume advertises frontend tools the live
            # (frozen) in-process client can't expose. The SDK freezes an in-process
            # server's tool list at connect() and the CLI refuses to hot-swap SDK
            # servers, so the only way to surface newly-activated tools within the
            # same user turn is a fresh resumed client. `tools_after_round` (iOS's
            # post-dispatch snapshot) is preferred over `input.tools`.
            advertised = _advertised_tools(input, resume_payload)
            widening = bool(frontend_qualified_names(advertised) - session.frontend_qualified)
            results = parse_tool_results(resume_payload)
            await session.resolve_results(results)
            if widening:
                # Deliver the gate result (above), then STOP the narrow turn before
                # it can loop re-calling the gate tool (it has no other tracker
                # tools to reach for). The interrupt yields a ResultMessage the pump
                # turns into a continuation turn with the widened surface attached.
                # If we can't cleanly interrupt, fall back to next-turn tools rather
                # than risk a stuck narrow turn.
                session.pending_widen_descriptors = advertised
                if not await _interrupt_for_widen(session):
                    session.pending_widen_descriptors = None
            return _stream(session)

        # --- Permission reply: the user is answering a parked approval -----
        # Only when the question actually reached them. An ask raised while the
        # session was held for background work sits in the backlog unseen, and
        # reading the next message as its yes/no would consume a message the user
        # meant as a new request — see `gate._ask_user`.
        parked = registry.get(thread_id)
        if parked is not None and not parked.disposed and parked.pending_decision is not None \
                and not parked.pending_decision.done() and not parked.pending_decision_delivered:
            logger.info(
                "claude_code loop: denying an approval the user never saw — their "
                "message starts a new turn instead (thread=%s)", thread_id,
            )
            undelivered = parked.pending_decision
            parked.pending_decision = None
            undelivered.set_result(False)  # fail-closed; the ask text rides this run
        if parked is not None and not parked.disposed and parked.pending_decision is not None \
                and not parked.pending_decision.done():
            reply = _latest_user_text(input.messages)
            allow = interpret_approval(reply)
            if allow and interpret_always(reply):
                parked.auto_approve = True  # run freely for the rest of this thread
            parked.open_run(run_id, events.run_started(thread_id, run_id))
            fut = parked.pending_decision
            parked.pending_decision = None
            fut.set_result(allow)
            return _stream(parked)

        # --- New-turn POST: build the session and start the pump ---------------
        # Evict whatever other threads abandoned, on the same lazy-sweep idiom
        # as the replay log: nothing else ever reaps `_REGISTRY`, so a client
        # that never came back left its session — and its SDK transport — held
        # for the life of the process. A parked turn is bounded far below the
        # idle timeout and a backgrounded client keeps touching its own session,
        # so anything this old is genuinely gone.
        await registry.sweep_idle()

        # A session parked on in-flight background work must not be retired: its
        # CLI child owns those tasks, so a fresh subprocess would orphan them.
        # Feed this turn into the live client instead, when it can serve the
        # request (see `_cannot_continue_live`).
        inherited: list = []
        live = registry.get(thread_id)
        if live is not None and not live.disposed and live.background.holding:
            declined = _cannot_continue_live(live, input)
            if declined is None:
                continued = await _continue_live_turn(live, input, run_id, mcp)
                if continued is not None:
                    return continued
                declined = "the query could not be sent"
            logger.info(
                "claude_code loop: cannot continue the live session (%s) — "
                "starting fresh; %s will be orphaned (thread=%s)",
                declined, live.background.summary(), thread_id,
            )
            # The background output already collected is the whole point of
            # holding the session; hand it to the replacement rather than
            # dropping it with the session that gathered it.
            inherited = live.take_deferred()

        # Any session still on this thread is parked mid-tool-call (the app went
        # away and came back with something new). Wind it down cleanly first —
        # closing its transport from under an in-flight hook leaves the SDK
        # session interrupted, and the next resume answers a synthetic no-op
        # instead of this prompt. Bounded; see `registry.retire`.
        await registry.retire(thread_id)
        session = registry.create(thread_id)
        # Stash the turn's request + shared MCP so the pump can rebuild options for
        # a continuation turn (gate unlock) without this request in scope.
        session.turn_input = input
        session.turn_mcp = mcp
        session.deferred.extend(inherited)
        session.open_run(run_id, events.run_started(thread_id, run_id))
        try:
            options = _options_for(
                input, session, mcp, list(input.tools or []),
                registry.remembered_session_id(thread_id),
            )
            client = ClaudeSDKClient(options)
            await client.connect()
            session.client = client
            has_prior = registry.remembered_session_id(thread_id) is not None
            latest_user = _latest_user_message(input.messages)
            image_blocks = _image_blocks(_msg_content(latest_user))
            # Resume turns send only the newest user message (the SDK replays prior
            # context via the session); first turns flatten the whole transcript.
            text = _coerce_content(_msg_content(latest_user)) if has_prior \
                else _render_transcript(input.messages)
            await _send_query(client, _query_content(text, image_blocks))
        except Exception as exc:  # noqa: BLE001
            logger.exception("claude_code loop: failed to start turn")
            await registry.remove(thread_id)
            return _error_stream(f"claude_code loop failed to start: {exc}")

        import asyncio

        session.pump_task = asyncio.ensure_future(_pump(session))
        return _stream(session)
