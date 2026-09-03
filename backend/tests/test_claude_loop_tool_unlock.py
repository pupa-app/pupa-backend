"""Mid-turn tool-unlock for the Claude Code loop — continuation-turn approach.

An in-process SDK MCP server's tool list is frozen at `connect()` (the SDK
advertises the tools capability *without* `listChanged`, and the CLI refuses
`mcp_toggle` for SDK servers — `"SDK servers should be handled in print.ts"`), so
a gate tool that unlocks more tools (e.g. `get_tools_tracker`) cannot widen the
live client. Instead the endpoint runs a **continuation turn**: when the current
turn finishes and the resume advertised tools the live client didn't have, it
builds a fresh `ClaudeSDKClient` exposing the widened set, `resume`s the same SDK
session, queries a synthetic continue prompt, and streams it onto the same SSE.

These tests pin that hand-off end-to-end (against a fake SDK client — the real
CLI path is exercised by a separate manual check), plus the early session-id
capture that keeps context alive when a turn errors before its ResultMessage.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
from fastapi import FastAPI

from pupa_backend.harnesses.claude import endpoint as cl_endpoint
from pupa_backend.harnesses.claude import env as cl_env
from pupa_backend.harnesses.claude import registry
from pupa_backend.harnesses.claude.frontend_tools import SERVER_NAME, frontend_qualified_names, qualified_name


# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #

def _tools(*names: str) -> list[dict]:
    return [
        {"name": n, "description": f"{n} desc", "parameters": {"type": "object", "properties": {}}}
        for n in names
    ]


async def _await_pending(thread_id: str, call_id: str) -> registry.LiveSession:
    """Spin until the pump has registered `call_id` as pending on the session."""
    for _ in range(1_000_000):
        sess = registry.get(thread_id)
        if sess and call_id in sess.pending:
            return sess
        await asyncio.sleep(0)
    raise AssertionError("pending call never registered")


class _FakeSDKClient:
    """Fake `ClaudeSDKClient` that records every constructed instance + its
    options, and branches behaviour on the frontend tools it was given.

    - Given ONLY `get_tools_tracker` → the "gate" turn: emit the gate tool call
      (→ interrupt), park until the resume resolves it, then finish with a
      session id.
    - Given `renderTracker` (the widened set) → the "continuation" turn: emit a
      short confirmation + finish. Its construction args are what the test asserts
      (resume id + widened allowed_tools + the continuation prompt).
    """

    instances: list["_FakeSDKClient"] = []

    def __init__(self, options=None, transport=None):
        self.options = options
        self.thread_id = "thread-gate"
        self.queries: list[str] = []
        self.disconnected = False
        self.interrupted = False
        _FakeSDKClient.instances.append(self)

    @property
    def _is_continuation(self) -> bool:
        return qualified_name("renderTracker") in (self.options.allowed_tools or [])

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self.queries.append(prompt)

    async def interrupt(self):
        self.interrupted = True

    async def disconnect(self):
        self.disconnected = True

    async def receive_messages(self):
        if self._is_continuation:
            yield AssistantMessage(
                content=[TextBlock(text="Tracker ready — added your rows.")],
                model="fake",
                message_id="m-cont",
                session_id="sdk-sess-1",
            )
            yield ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1,
                is_error=False, num_turns=1, session_id="sdk-sess-1",
            )
            return
        # Gate turn.
        yield AssistantMessage(
            content=[ToolUseBlock(id="call-gate", name=qualified_name("get_tools_tracker"), input={})],
            model="fake",
            message_id="m-gate",
            session_id="sdk-sess-1",
        )
        sess = await _await_pending(self.thread_id, "call-gate")
        await sess.claim_call("get_tools_tracker", {})  # blocks until the resume resolves it
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1,
            is_error=False, num_turns=1, session_id="sdk-sess-1",
        )


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def loop_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _FakeSDKClient)
    _FakeSDKClient.instances = []
    registry._REGISTRY.clear()
    registry._SESSION_IDS.clear()
    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)
    return app


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_advertised_tools_prefers_tools_after_round() -> None:
    from ag_ui.core import RunAgentInput

    input = RunAgentInput.model_validate({
        "thread_id": "t", "run_id": "r", "messages": [],
        "tools": _tools("get_tools_tracker"), "state": {}, "context": [],
        "forwardedProps": {},
    })
    resume = {"tool_results": [], "tools_after_round": _tools("get_tools_tracker", "renderTracker")}
    # `frontend_qualified_names` handles both dicts and pydantic Tool descriptors,
    # so assert on it rather than indexing raw entries.
    assert frontend_qualified_names(cl_endpoint._advertised_tools(input, resume)) == {
        qualified_name("get_tools_tracker"), qualified_name("renderTracker"),
    }
    # Falls back to input.tools when tools_after_round is absent/empty.
    assert frontend_qualified_names(cl_endpoint._advertised_tools(input, {"tool_results": []})) == {
        qualified_name("get_tools_tracker"),
    }


def test_frontend_qualified_names_no_server_build() -> None:
    assert frontend_qualified_names(_tools("a", "b")) == {qualified_name("a"), qualified_name("b")}
    assert frontend_qualified_names([]) == set()


# --------------------------------------------------------------------------- #
# End-to-end: a gate-widening resume runs a continuation turn
# --------------------------------------------------------------------------- #

async def test_gate_widening_resume_runs_continuation(loop_app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # POST 1: only the gate tool advertised; the model calls it → interrupt.
        r1 = await client.post("/", json={
            "thread_id": "thread-gate", "run_id": "run-1",
            "messages": [{"id": "u1", "role": "user", "content": "track my books"}],
            "tools": _tools("get_tools_tracker"), "state": {}, "context": [], "forwardedProps": {},
        })
        assert any(e["type"] == "CUSTOM" for e in _sse_events(r1.text))
        session = registry.get("thread-gate")
        assert session is not None
        assert session.frontend_qualified == {qualified_name("get_tools_tracker")}
        # Bug A: the session id is captured from the assistant message, not just
        # the final result — so it's remembered even mid-turn.
        assert registry.remembered_session_id("thread-gate") == "sdk-sess-1"

        # POST 2 (resume): device activated the tracker; the widened surface now
        # includes renderTracker. The turn finishes → a continuation turn runs.
        r2 = await client.post("/", json={
            "thread_id": "thread-gate", "run_id": "run-2", "messages": [],
            "tools": _tools("get_tools_tracker", "renderTracker", "addTrackerItems"),
            "state": {}, "context": [],
            "forwardedProps": {"command": {"resume": {
                "tool_results": [{"toolCallId": "call-gate", "content": "activated"}],
                "tools_after_round": _tools("get_tools_tracker", "renderTracker", "addTrackerItems"),
            }}},
        })
        types = [e["type"] for e in _sse_events(r2.text)]
        assert "RUN_FINISHED" in types

    # A second client was built for the continuation turn: resumes the SAME SDK
    # session and exposes the widened tools; the first client was torn down.
    assert len(_FakeSDKClient.instances) == 2, [c.options.allowed_tools for c in _FakeSDKClient.instances]
    gate, cont = _FakeSDKClient.instances
    assert cont.options.resume == "sdk-sess-1"
    assert qualified_name("renderTracker") in cont.options.allowed_tools
    assert qualified_name("addTrackerItems") in cont.options.allowed_tools
    assert cont.queries == [cl_endpoint._CONTINUATION_PROMPT]
    # The narrow gate turn was interrupted (so it couldn't loop re-calling the
    # gate) and then torn down.
    assert gate.interrupted is True
    assert gate.disconnected is True


async def test_non_widening_resume_does_not_continue(loop_app: FastAPI) -> None:
    """A resume that re-advertises the same surface must NOT spawn a continuation."""
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json={
            "thread_id": "thread-gate", "run_id": "run-1",
            "messages": [{"id": "u1", "role": "user", "content": "hi"}],
            "tools": _tools("get_tools_tracker"), "state": {}, "context": [], "forwardedProps": {},
        })
        # Resume advertises the SAME single tool → no widening.
        r2 = await client.post("/", json={
            "thread_id": "thread-gate", "run_id": "run-2", "messages": [],
            "tools": _tools("get_tools_tracker"), "state": {}, "context": [],
            "forwardedProps": {"command": {"resume": {
                "tool_results": [{"toolCallId": "call-gate", "content": "ok"}],
                "tools_after_round": _tools("get_tools_tracker"),
            }}},
        })
        assert "RUN_FINISHED" in [e["type"] for e in _sse_events(r2.text)]

    assert len(_FakeSDKClient.instances) == 1  # no continuation client built


# --------------------------------------------------------------------------- #
# Bug A: early session-id capture survives a turn that errors before the result
# --------------------------------------------------------------------------- #

async def test_session_id_remembered_when_turn_errors_before_result() -> None:
    """A turn that errors mid-stream still leaves a resumable session id so the
    next user message keeps prior context (iOS only re-sends user messages)."""
    registry._SESSION_IDS.clear()

    class _ErroringClient:
        async def receive_messages(self):
            yield AssistantMessage(
                content=[TextBlock(text="working…")],
                model="fake", message_id="m1", session_id="sdk-err-1",
            )
            raise RuntimeError("SDK blew up mid-turn")

    session = registry.LiveSession(thread_id="t-err")
    session.client = _ErroringClient()
    session.current_run_id = "run-x"

    await cl_endpoint._pump(session)

    assert registry.remembered_session_id("t-err") == "sdk-err-1"
    # The error was surfaced to the client (ERROR sentinel on the queue).
    drained = []
    while not session.queue.empty():
        drained.append(session.queue.get_nowait())
    assert registry.ERROR in drained
