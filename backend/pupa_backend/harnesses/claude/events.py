"""Translate `claude-agent-sdk` stream messages into AG-UI events.

The wire shapes here are matched against `ag_ui_langgraph` (the adapter the iOS
client already consumes):

- assistant text  → `TextMessageStart` / `TextMessageContent` / `TextMessageEnd`
- a tool call     → `ToolCallStart` / `ToolCallArgs` / `ToolCallEnd` (bare tool
                    name; the `mcp__pupa_frontend__` prefix is stripped)
- frontend batch  → one `CustomEvent(name="on_interrupt",
                    value={"frontend_tool_calls": [{id,name,args}, ...]})`
- lifecycle       → `RunStartedEvent` / `RunFinishedEvent` / `RunErrorEvent`

With `include_partial_messages` on, `receive_response()` also yields partial
`StreamEvent`s; `translate_stream_event` maps each text delta to a
`TextMessageContent`, so assistant text streams token-by-token. The whole
`AssistantMessage` still follows — the pump calls `translate_assistant_message`
with `skip_text=True` there so the already-streamed text isn't re-sent (tool
calls, which don't stream, are emitted from the whole message).

`skip_text` is decided **per message** via `text_already_streamed`, not set
blanket-true: the CLI fabricates some assistant messages locally (rate-limit
notices, API errors, the "No response requested." reply to a queued query) and
those never stream. Skipping them unconditionally deleted the only explanation
the user would get — the run emitted `RUN_STARTED` + `RUN_FINISHED` and nothing
else, which reads as a dropped connection.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ag_ui.core import EventType
from ag_ui.core.events import (
    CustomEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from claude_agent_sdk import ServerToolUseBlock, TextBlock, ToolUseBlock

from .frontend_tools import TOOL_PREFIX, bare_name

ON_INTERRUPT = "on_interrupt"


def run_started(thread_id: str, run_id: str) -> RunStartedEvent:
    return RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)


def run_finished(thread_id: str, run_id: str) -> RunFinishedEvent:
    return RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id)


def run_error(message: str, code: str | None = None) -> RunErrorEvent:
    return RunErrorEvent(type=EventType.RUN_ERROR, message=message, code=code)


def on_interrupt(frontend_calls: list[dict[str, Any]]) -> CustomEvent:
    """Build the batched `on_interrupt` event the iOS client dispatches locally."""
    return CustomEvent(
        type=EventType.CUSTOM,
        name=ON_INTERRUPT,
        value={"frontend_tool_calls": frontend_calls},
    )


def _text_events(message_id: str, text: str) -> list[Any]:
    return [
        TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant"),
        TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=text),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id),
    ]


def _tool_call_events(call_id: str, name: str, args: Any, parent_message_id: str | None) -> list[Any]:
    try:
        delta = json.dumps(args or {}, default=str)
    except (TypeError, ValueError):
        delta = "{}"
    return [
        ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=call_id,
            tool_call_name=name,
            parent_message_id=parent_message_id,
        ),
        ToolCallArgsEvent(type=EventType.TOOL_CALL_ARGS, tool_call_id=call_id, delta=delta),
        ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=call_id),
    ]


def new_stream_state() -> dict[str, Any]:
    """Fresh per-turn cursor for `translate_stream_event` / `text_already_streamed`.

    `streamed_text_ids` records every message id that opened a *text* block, so the
    pump can skip re-emitting exactly those and no others.
    """
    return {
        "message_id": None,
        "text_open": False,
        "streamed_text_ids": set(),
        "usage_logged_ids": set(),
    }


def record_usage_logged(msg: Any, state: dict[str, Any]) -> bool:
    """True the first time this turn sees `msg`'s message id — else False.

    The SDK emits one `AssistantMessage` per content block and every one carries
    the same message-level `usage`, so the pump would log identical token lines
    once per block. A message with no id is always logged (it can't be deduped
    against anything).
    """
    message_id = getattr(msg, "message_id", None)
    if message_id is None:
        return True
    seen = state.setdefault("usage_logged_ids", set())
    if message_id in seen:
        return False
    seen.add(message_id)
    return True


def text_already_streamed(msg: Any, state: dict[str, Any]) -> bool:
    """True when this `AssistantMessage`'s text reached the client as deltas.

    Drives the pump's `skip_text`. False for anything the CLI fabricated locally
    (no partial events, and usually `model="<synthetic>"`) and false for a message
    with no id, which must never collide with an unrecorded one.
    """
    message_id = getattr(msg, "message_id", None)
    if message_id is None:
        return False
    return message_id in state.get("streamed_text_ids", ())


def translate_stream_event(raw_event: dict[str, Any], state: dict[str, Any]) -> list[Any]:
    """Map one raw Anthropic partial-message stream event → AG-UI text events.

    Called per `StreamEvent` when `include_partial_messages` is on, so assistant
    **text** streams as incremental `TextMessageContent` deltas instead of one
    buffered block. `state` (see `new_stream_state`) is mutated across calls within
    a turn: `message_start` records the id; a text `content_block_start` opens a
    start/…/end run and marks the id as streamed; `content_block_stop` closes it.

    Text only — `thinking_delta`s stay hidden, and tool calls are dispatched from
    the whole `AssistantMessage` (frontend dispatch needs complete args JSON, which
    partials don't provide until block-stop). So everything non-text returns `[]`.
    """
    etype = raw_event.get("type")
    if etype == "message_start":
        state["message_id"] = (raw_event.get("message") or {}).get("id")
        state["text_open"] = False
        return []
    if etype == "content_block_start":
        if (raw_event.get("content_block") or {}).get("type") == "text":
            state["text_open"] = True
            if state.get("message_id") is not None:
                state.setdefault("streamed_text_ids", set()).add(state["message_id"])
            return [
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=state["message_id"],
                    role="assistant",
                )
            ]
        return []
    if etype == "content_block_delta":
        delta = raw_event.get("delta") or {}
        if delta.get("type") == "text_delta" and delta.get("text"):
            return [
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=state["message_id"],
                    delta=delta["text"],
                )
            ]
        return []
    if etype == "content_block_stop" and state.get("text_open"):
        state["text_open"] = False
        return [
            TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=state["message_id"])
        ]
    return []


def translate_assistant_message(
    msg: Any, skip_text: bool = False
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Turn one `AssistantMessage` into (AG-UI events, frontend_calls).

    `frontend_calls` is the list of `{id, name, args}` for tool calls that target
    on-device frontend tools — the endpoint registers a pending result slot for each
    and appends one batched `on_interrupt`.

    We emit `ToolCall*` UI events for **all** tool calls, but only frontend tools are
    added to `frontend_calls`. Native host tools (`Read`, `Bash`, `Grep`, …) and
    server tools (`ServerToolUseBlock`: web_search / web_fetch / …) run in-process
    with no iOS handler, so they are surfaced **display-only** — start/args/end so the
    client shows a tool bubble, but never added to `frontend_calls`. That keeps them
    off the on_interrupt/dispatch/resume path, so there is no pending slot and no
    dangling call to fulfil. `ThinkingBlock`s are still skipped.

    `skip_text=True` omits the text events — used when tokens already streamed live
    via `translate_stream_event`, so the whole-message text isn't re-sent. Tool calls
    are unaffected (they're never streamed).
    """
    events: list[Any] = []
    frontend_calls: list[dict[str, Any]] = []
    message_id = getattr(msg, "message_id", None) or str(uuid.uuid4())

    for block in getattr(msg, "content", []) or []:
        if isinstance(block, TextBlock):
            if block.text and not skip_text:
                events.extend(_text_events(message_id, block.text))
        elif isinstance(block, ToolUseBlock):
            if block.name.startswith(TOOL_PREFIX):
                # Frontend tool: emit UI events AND park for on-device dispatch.
                display_name = bare_name(block.name)
                events.extend(_tool_call_events(block.id, display_name, block.input, message_id))
                frontend_calls.append(
                    {"id": block.id, "name": display_name, "args": block.input or {}}
                )
            else:
                # Native host tool (Read/Bash/Grep/…): display-only bubble.
                # Emit UI events but do NOT add to frontend_calls — no dispatch.
                events.extend(_tool_call_events(block.id, block.name, block.input, message_id))
        elif isinstance(block, ServerToolUseBlock):
            # Server tool (web_search / web_fetch / …): display-only bubble.
            events.extend(_tool_call_events(block.id, block.name, block.input, message_id))
    return events, frontend_calls
