"""Tests for GET /db/threads/{thread_id}/messages — normalized transcript endpoint.

Drives a short graph run through a MemorySaver, then asserts
the endpoint returns the human/ai/tool sequence in order with tool calls attached.
"""



import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from pupa_backend.harnesses.langgraph.db.routes import router as db_router

from .conftest import MockChatModel


# Per-route scope enforcement on /db/threads/* requires either ``api_key``
# identity or a device with the ``agent`` scope. These tests target the
# route handler shape, not auth — disable auth at the module level so the
# scope dependency short-circuits without needing to mount the middleware.
@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")


def _make_app(checkpointer: BaseCheckpointSaver) -> FastAPI:
    app = FastAPI()
    app.state.checkpointer = checkpointer
    app.include_router(db_router, prefix="/db")
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@tool
def dummy_tool(value: str) -> str:  # noqa: D103
    """A trivial tool so create_agent installs a ToolNode."""
    return f"result:{value}"


def _ai_with_tool_call(text: str = "") -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[{"name": "dummy_tool", "args": {"value": "ping"}, "id": "tc-1"}],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_thread_returns_empty_list() -> None:
    """A thread with no checkpoint yet returns [] — not 404."""
    app = _make_app(MemorySaver())
    with TestClient(app) as client:
        resp = client.get("/db/threads/nonexistent-thread/messages")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_simple_human_ai_exchange() -> None:
    """A two-turn exchange appears as [human, ai] in the right order."""
    thread_id = "test-simple-exchange"
    config = {"configurable": {"thread_id": thread_id}}
    cp = MemorySaver()

    model = MockChatModel(responses=[AIMessage(content="Hello!")])
    graph = create_agent(model, checkpointer=cp)
    await graph.ainvoke({"messages": [HumanMessage(content="Hi")]}, config=config)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.get(f"/db/threads/{thread_id}/messages")

    assert resp.status_code == 200
    msgs = resp.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "human"
    assert msgs[0]["content"] == "Hi"
    assert msgs[1]["role"] == "ai"
    assert msgs[1]["content"] == "Hello!"
    assert msgs[1]["tool_calls"] == []


@pytest.mark.asyncio
async def test_tool_call_round_attached_to_ai_message() -> None:
    """An AI message with tool_calls is followed by the tool result message."""
    thread_id = "test-tool-round"
    config = {"configurable": {"thread_id": thread_id}}
    cp = MemorySaver()

    # Round 1: model emits a tool call; round 2: model settles with text.
    model = MockChatModel(
        responses=[
            _ai_with_tool_call(),
            AIMessage(content="Done!"),
        ]
    )
    graph = create_agent(model, tools=[dummy_tool], checkpointer=cp)
    await graph.ainvoke({"messages": [HumanMessage(content="Use the tool")]}, config=config)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.get(f"/db/threads/{thread_id}/messages")

    assert resp.status_code == 200
    msgs = resp.json()

    roles = [m["role"] for m in msgs]
    # Expect: human, ai (with tool_calls), tool, ai (final text)
    assert roles[0] == "human"
    ai_with_calls = next(m for m in msgs if m["role"] == "ai" and m["tool_calls"])
    assert ai_with_calls["tool_calls"][0]["name"] == "dummy_tool"
    assert ai_with_calls["tool_calls"][0]["id"] == "tc-1"
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert "result:ping" in tool_msg["content"]
    assert tool_msg["tool_call_id"] == "tc-1"
    final_ai = msgs[-1]
    assert final_ai["role"] == "ai"
    assert final_ai["content"] == "Done!"


@pytest.mark.asyncio
async def test_multi_turn_preserves_order() -> None:
    """Two separate user turns accumulate and come back in chronological order."""
    thread_id = "test-multi-turn"
    config = {"configurable": {"thread_id": thread_id}}
    cp = MemorySaver()

    model = MockChatModel(
        responses=[
            AIMessage(content="Turn 1 reply"),
            AIMessage(content="Turn 2 reply"),
        ]
    )
    graph = create_agent(model, checkpointer=cp)
    await graph.ainvoke({"messages": [HumanMessage(content="First")]}, config=config)
    await graph.ainvoke({"messages": [HumanMessage(content="Second")]}, config=config)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.get(f"/db/threads/{thread_id}/messages")

    assert resp.status_code == 200
    msgs = resp.json()
    roles = [m["role"] for m in msgs]
    assert roles == ["human", "ai", "human", "ai"]
    assert msgs[0]["content"] == "First"
    assert msgs[2]["content"] == "Second"
