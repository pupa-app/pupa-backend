"""Harness-independent AG-UI event builders.

Harnesses translate different model runtimes into the same small set of AG-UI
frames.  Keeping the constructors here prevents their wire shapes from drifting
between implementations.
"""

from __future__ import annotations

import json
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

ON_INTERRUPT = "on_interrupt"


def run_started(thread_id: str, run_id: str) -> RunStartedEvent:
    return RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)


def run_finished(thread_id: str, run_id: str) -> RunFinishedEvent:
    return RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id)


def run_error(message: str, code: str | None = None) -> RunErrorEvent:
    return RunErrorEvent(type=EventType.RUN_ERROR, message=message, code=code)


def on_interrupt(frontend_calls: list[dict[str, Any]]) -> CustomEvent:
    return CustomEvent(
        type=EventType.CUSTOM,
        name=ON_INTERRUPT,
        value={"frontend_tool_calls": frontend_calls},
    )


def text_events(message_id: str, text: str) -> list[Any]:
    return [
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role="assistant",
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=message_id,
            delta=text,
        ),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id),
    ]


def tool_call_events(
    call_id: str,
    name: str,
    args: Any,
    parent_message_id: str | None = None,
) -> list[Any]:
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
