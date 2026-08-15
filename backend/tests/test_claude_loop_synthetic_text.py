"""Non-streamed assistant text must still reach the client.

The pump asks for `skip_text` because `translate_stream_event` already streamed
the tokens. But the CLI fabricates some assistant messages locally — rate-limit
notices ("You've hit your session limit"), API errors, and the
"No response requested." reply it emits when a `query()` was queued behind a
resumed session's "Continue from where you left off." — and those never produce
partial `StreamEvent`s. Skipping their text unconditionally deleted the only
explanation the user would ever get: the turn emitted `RUN_STARTED` +
`RUN_FINISHED` and nothing else, which reads as a dropped connection.

So `skip_text` must be decided **per message**: skip only the text of a message
whose id actually streamed.
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
)
from fastapi import FastAPI

from pupa_backend.harnesses.claude.events import (
    new_stream_state,
    text_already_streamed,
    translate_stream_event,
)


def _drain(raw_events, state):
    out = []
    for raw in raw_events:
        out.extend(translate_stream_event(raw, state))
    return out


def _synthetic(text: str, message_id: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)], model="<synthetic>", message_id=message_id
    )


# --------------------------------------------------------------------------- #
# Unit: which messages count as already-streamed
# --------------------------------------------------------------------------- #


def test_streamed_message_is_marked_as_streamed() -> None:
    state = new_stream_state()
    _drain(
        [
            {"type": "message_start", "message": {"id": "msg-1"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_stop", "index": 0},
        ],
        state,
    )

    assert text_already_streamed(AssistantMessage(content=[], model="m", message_id="msg-1"), state)


def test_synthetic_message_is_not_marked_as_streamed() -> None:
    """A locally-fabricated message never streamed, so its text must be emitted."""
    state = new_stream_state()
    _drain(
        [
            {"type": "message_start", "message": {"id": "msg-1"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_stop", "index": 0},
        ],
        state,
    )

    assert not text_already_streamed(_synthetic("You've hit your session limit", "msg-synth"), state)


def test_message_with_no_id_is_never_treated_as_streamed() -> None:
    """`message_id=None` must not collide with an unrecorded id and get skipped."""
    state = new_stream_state()
    _drain([{"type": "message_start", "message": {}}], state)

    assert not text_already_streamed(_synthetic("No response requested."), state)


def test_tool_only_stream_does_not_mark_text_as_streamed() -> None:
    """A message whose only streamed block was a tool call still needs its text."""
    state = new_stream_state()
    _drain(
        [
            {"type": "message_start", "message": {"id": "msg-2"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "Read"}},
            {"type": "content_block_stop", "index": 0},
        ],
        state,
    )

    assert not text_already_streamed(_synthetic("late text", "msg-2"), state)


# --------------------------------------------------------------------------- #
# Endpoint: the real regression — a turn that only produces a synthetic message
# --------------------------------------------------------------------------- #


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


_LIMIT_NOTICE = "You've hit your session limit · resets 7:10pm (Europe/London)"


class _SyntheticOnlySDKClient:
    """Fake client for the wake-up failure: the resumed session answers with one
    locally-fabricated assistant message and no partial StreamEvents at all."""

    def __init__(self, options=None, transport=None):
        self.options = options

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self._prompt = prompt

    async def disconnect(self):
        return None

    async def receive_response(self):
        yield _synthetic(_LIMIT_NOTICE, "msg-synth")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-s",
        )


class _MixedSDKClient:
    """Streams real text for one message, then appends a synthetic notice."""

    def __init__(self, options=None, transport=None):
        self.options = options

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self._prompt = prompt

    async def disconnect(self):
        return None

    async def receive_response(self):
        def ev(payload):
            return StreamEvent(uuid="x", session_id="sdk-s", event=payload)

        yield ev({"type": "message_start", "message": {"id": "msg-real"}})
        yield ev({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        yield ev({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Working"}})
        yield ev({"type": "content_block_stop", "index": 0})
        yield AssistantMessage(content=[TextBlock(text="Working")], model="fake", message_id="msg-real")
        yield _synthetic(_LIMIT_NOTICE, "msg-synth")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-s",
        )


def _app(monkeypatch: pytest.MonkeyPatch, client_cls) -> FastAPI:
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint
    from pupa_backend.harnesses.claude import env as cl_env

    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"}
    )
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", client_cls)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)
    return app


async def _run(app: FastAPI, thread_id: str) -> list[dict]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/",
            json={
                "thread_id": thread_id,
                "run_id": "r1",
                "messages": [{"id": "u1", "role": "user", "content": "hi"}],
                "tools": [],
                "state": {},
                "context": [],
                "forwardedProps": {},
            },
        )
    return _sse_events(r.text)


async def test_synthetic_only_turn_surfaces_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wake-up bug: without this the run was RUN_STARTED + RUN_FINISHED only."""
    evs = await _run(_app(monkeypatch, _SyntheticOnlySDKClient), "synthetic-thread")

    types = [e["type"] for e in evs]
    deltas = [e["delta"] for e in evs if e["type"] == "TEXT_MESSAGE_CONTENT"]

    assert deltas == [_LIMIT_NOTICE]
    assert types.count("TEXT_MESSAGE_START") == 1
    assert types.count("TEXT_MESSAGE_END") == 1
    assert "RUN_FINISHED" in types


async def test_streamed_text_is_not_doubled_when_a_synthetic_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamed text stays single-emitted; only the synthetic message adds text."""
    evs = await _run(_app(monkeypatch, _MixedSDKClient), "mixed-thread")

    deltas = [e["delta"] for e in evs if e["type"] == "TEXT_MESSAGE_CONTENT"]

    assert deltas == ["Working", _LIMIT_NOTICE]


def test_translate_stream_event_accepts_a_bare_state_dict() -> None:
    """Back-compat: callers that pass the old `{message_id, text_open}` dict still
    work — `new_stream_state` only adds a key."""
    state: dict = {"message_id": None, "text_open": False}
    events = _drain(
        [
            {"type": "message_start", "message": {"id": "msg-3"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_stop", "index": 0},
        ],
        state,
    )

    assert [e.type for e in events] == [
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_END,
    ]
    assert text_already_streamed(
        AssistantMessage(content=[], model="m", message_id="msg-3"), state
    )
