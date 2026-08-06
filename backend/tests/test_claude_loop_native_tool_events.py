"""`translate_assistant_message` surfaces native + server tool calls as
display-only tool bubbles (start/args/end) without adding them to
`frontend_calls` — so they never enter the on_interrupt/dispatch/resume path.
Frontend tools still both emit events AND park for dispatch.
"""

from __future__ import annotations

import json

from ag_ui.core import EventType
from claude_agent_sdk import (
    AssistantMessage,
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
)

from pupa_backend.harnesses.claude import frontend_tools
from pupa_backend.harnesses.claude.events import translate_assistant_message


def _types(events):
    return [e.type for e in events]


def test_native_and_server_tools_emit_display_only_events() -> None:
    msg = AssistantMessage(
        content=[
            TextBlock(text="working on it"),
            ToolUseBlock(id="call-read", name="Read", input={"file_path": "/x"}),
            ServerToolUseBlock(id="call-web", name="web_search", input={"query": "q"}),
            ToolUseBlock(
                id="call-fe",
                name=frontend_tools.qualified_name("renderChecklist"),
                input={"items": ["a"]},
            ),
        ],
        model="fake",
        message_id="m-1",
    )

    events, frontend_calls = translate_assistant_message(msg)

    # Only the frontend tool parks for dispatch.
    assert [c["id"] for c in frontend_calls] == ["call-fe"]
    assert frontend_calls[0]["name"] == "renderChecklist"  # bare name

    # Every tool call (native, server, frontend) got a start/args/end triple.
    starts = {
        e.tool_call_id: e.tool_call_name
        for e in events
        if e.type == EventType.TOOL_CALL_START
    }
    assert starts == {
        "call-read": "Read",       # bare native name, unprefixed
        "call-web": "web_search",  # server tool name
        "call-fe": "renderChecklist",
    }

    for cid in ("call-read", "call-web", "call-fe"):
        ids_for = [
            e.type
            for e in events
            if getattr(e, "tool_call_id", None) == cid
        ]
        assert ids_for == [
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
        ]

    # Args are JSON-serialised into the ToolCallArgs delta.
    read_args = next(
        e
        for e in events
        if e.type == EventType.TOOL_CALL_ARGS and e.tool_call_id == "call-read"
    )
    assert json.loads(read_args.delta) == {"file_path": "/x"}


def test_native_only_message_produces_no_frontend_calls() -> None:
    msg = AssistantMessage(
        content=[ToolUseBlock(id="c1", name="Bash", input={"command": "ls"})],
        model="fake",
        message_id="m-2",
    )

    events, frontend_calls = translate_assistant_message(msg)

    assert frontend_calls == []
    assert EventType.TOOL_CALL_START in _types(events)
