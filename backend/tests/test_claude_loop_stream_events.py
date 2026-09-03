"""Token-level streaming for the claude_code loop.

`translate_stream_event` maps raw Anthropic partial-message stream events
(`message_start` / `content_block_*`) into incremental AG-UI text events, so a
single assistant reply arrives as many `TEXT_MESSAGE_CONTENT` deltas instead of
one buffered chunk. `translate_assistant_message(skip_text=True)` then suppresses
the whole-message text (already streamed) while keeping tool calls intact.
"""

from __future__ import annotations

import json

import httpx
import pytest
from ag_ui.core import EventType
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
)
from fastapi import FastAPI

from pupa_backend.harnesses.claude import frontend_tools
from pupa_backend.harnesses.claude.events import (
    translate_assistant_message,
    translate_stream_event,
)


def _types(events):
    return [e.type for e in events]


def _drain(raw_events):
    """Feed a sequence of raw stream-event dicts through one shared state."""
    state: dict = {"message_id": None, "text_open": False}
    out = []
    for raw in raw_events:
        out.extend(translate_stream_event(raw, state))
    return out


def test_text_deltas_stream_as_incremental_content_events() -> None:
    events = _drain(
        [
            {"type": "message_start", "message": {"id": "msg-1"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
            {"type": "content_block_stop", "index": 0},
        ]
    )

    assert _types(events) == [
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
    ]
    # One message_id threads the whole block.
    assert {e.message_id for e in events} == {"msg-1"}
    # Deltas concatenate to the full text — client accumulates them.
    deltas = [e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    assert "".join(deltas) == "Hello world"


def test_thinking_and_tool_stream_events_emit_no_text() -> None:
    events = _drain(
        [
            {"type": "message_start", "message": {"id": "msg-2"}},
            # thinking stays hidden per scope
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_stop", "index": 0},
            # tool_use is dispatched from the whole message, not streamed
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "Read"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
    )

    assert events == []


def test_skip_text_suppresses_message_text_but_keeps_tool_calls() -> None:
    msg = AssistantMessage(
        content=[
            TextBlock(text="already streamed"),
            ToolUseBlock(id="call-read", name="Read", input={"file_path": "/x"}),
            ToolUseBlock(
                id="call-fe",
                name=frontend_tools.qualified_name("renderChecklist"),
                input={"items": ["a"]},
            ),
        ],
        model="fake",
        message_id="m-1",
    )

    events, frontend_calls = translate_assistant_message(msg, skip_text=True)

    # Text was streamed already — no text events on the whole-message path.
    assert EventType.TEXT_MESSAGE_START not in _types(events)
    assert EventType.TEXT_MESSAGE_CONTENT not in _types(events)
    assert EventType.TEXT_MESSAGE_END not in _types(events)

    # Tool calls still surface, and only the frontend tool parks for dispatch.
    starts = {
        e.tool_call_id: e.tool_call_name
        for e in events
        if e.type == EventType.TOOL_CALL_START
    }
    assert starts == {"call-read": "Read", "call-fe": "renderChecklist"}
    assert [c["id"] for c in frontend_calls] == ["call-fe"]


def test_default_still_emits_whole_message_text() -> None:
    """Regression guard: without skip_text the whole-block text is still emitted."""
    msg = AssistantMessage(
        content=[TextBlock(text="hi")],
        model="fake",
        message_id="m-2",
    )

    events, _ = translate_assistant_message(msg)

    assert _types(events) == [
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
    ]


# --------------------------------------------------------------------------- #
# Endpoint wiring: partial StreamEvents stream through the pump as deltas, and
# the whole AssistantMessage that follows does NOT re-emit the same text.
# --------------------------------------------------------------------------- #

def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


class _StreamingSDKClient:
    """Fake `ClaudeSDKClient` that emits partial text StreamEvents, then the whole
    `AssistantMessage` carrying the same text, then the terminal result."""

    def __init__(self, options=None, transport=None):
        self.options = options

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self._prompt = prompt

    async def disconnect(self):
        return None

    async def receive_messages(self):
        def ev(payload):
            return StreamEvent(uuid="x", session_id="sdk-s", event=payload)

        yield ev({"type": "message_start", "message": {"id": "msg-x"}})
        yield ev({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        yield ev({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}})
        yield ev({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}})
        yield ev({"type": "content_block_stop", "index": 0})
        # The SDK still delivers the assembled whole message after the partials.
        yield AssistantMessage(content=[TextBlock(text="Hi there")], model="fake", message_id="msg-x")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-s",
        )


async def test_endpoint_streams_text_deltas_without_double_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint
    from pupa_backend.harnesses.claude import env as cl_env

    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _StreamingSDKClient)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "thread_id": "stream-thread",
            "run_id": "r1",
            "messages": [{"id": "u1", "role": "user", "content": "hi"}],
            "tools": [],
            "state": {},
            "context": [],
            "forwardedProps": {},
        }
        r = await client.post("/", json=body)

    evs = _sse_events(r.text)
    types = [e["type"] for e in evs]
    deltas = [e["delta"] for e in evs if e["type"] == "TEXT_MESSAGE_CONTENT"]

    # Text arrived as two incremental deltas — not one buffered "Hi there" chunk,
    # and not doubled by the trailing whole message.
    assert deltas == ["Hi", " there"]
    assert types.count("TEXT_MESSAGE_START") == 1
    assert types.count("TEXT_MESSAGE_END") == 1
    assert "RUN_FINISHED" in types
