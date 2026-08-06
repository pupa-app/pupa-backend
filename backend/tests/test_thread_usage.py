"""Tests for POST /db/threads/usage — batched Langfuse-backed token + cost.

Drives short graph runs through a MemorySaver to create real
checkpoints (fingerprints), stubs ``fetch_usage`` so no Langfuse is touched, and
asserts the batch shape, the fingerprint cache, and the Langfuse-disabled path.
"""



import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

import pupa_backend.harnesses.langgraph.db.routes as routes_mod
from pupa_backend.harnesses.langgraph.observability.usage import CacheRow, UsageRow
from pupa_backend.harnesses.langgraph.db.routes import router as db_router

from .conftest import MockChatModel


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")


@pytest.fixture(autouse=True)
def _clear_usage_cache() -> None:
    """Usage + cache caches are module-global; reset around every test."""
    for d in (routes_mod._USAGE_CACHE, routes_mod._USAGE_FETCHED_AT,
              routes_mod._CACHE_USAGE, routes_mod._CACHE_FETCHED_AT):
        d.clear()
    yield
    for d in (routes_mod._USAGE_CACHE, routes_mod._USAGE_FETCHED_AT,
              routes_mod._CACHE_USAGE, routes_mod._CACHE_FETCHED_AT):
        d.clear()


async def _seed(cp: MemorySaver, thread_id: str) -> None:
    """Run one turn so the thread has a checkpoint (fingerprint)."""
    model = MockChatModel(responses=[AIMessage(content="Hello!")])
    graph = create_agent(model, checkpointer=cp)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": thread_id}},
    )


def _make_app(checkpointer: BaseCheckpointSaver) -> FastAPI:
    app = FastAPI()
    app.state.checkpointer = checkpointer
    app.include_router(db_router, prefix="/db")
    return app


@pytest.mark.asyncio
async def test_batch_merges_langfuse_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known thread gets Langfuse totals; thread with no trace gets nulls."""
    thread_id = "usage-known"
    cp = MemorySaver()
    model = MockChatModel(responses=[AIMessage(content="Hello!")])
    graph = create_agent(model, checkpointer=cp)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    async def _stub(session_ids):
        return {thread_id: UsageRow(total_tokens=1234, cost_usd=0.0456)}

    monkeypatch.setattr(routes_mod, "fetch_usage", _stub)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.post(
            "/db/threads/usage",
            json={"thread_ids": [thread_id, "no-such-thread"]},
        )

    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage[thread_id]["total_tokens"] == 1234
    assert usage[thread_id]["cost_usd"] == pytest.approx(0.0456)
    assert usage[thread_id]["fingerprint"] is not None
    # Unknown thread: no checkpoint, no Langfuse row → all null.
    assert usage["no-such-thread"]["total_tokens"] is None
    assert usage["no-such-thread"]["cost_usd"] is None
    assert usage["no-such-thread"]["fingerprint"] is None


@pytest.mark.asyncio
async def test_unchanged_fingerprint_serves_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second request with no new turn must not re-query Langfuse."""
    thread_id = "usage-cached"
    cp = MemorySaver()
    model = MockChatModel(responses=[AIMessage(content="Hello!")])
    graph = create_agent(model, checkpointer=cp)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    calls = {"n": 0}

    async def _stub(session_ids):
        calls["n"] += 1
        return {thread_id: UsageRow(total_tokens=10, cost_usd=0.01)}

    monkeypatch.setattr(routes_mod, "fetch_usage", _stub)

    app = _make_app(cp)
    with TestClient(app) as client:
        r1 = client.post("/db/threads/usage", json={"thread_ids": [thread_id]})
        r2 = client.post("/db/threads/usage", json={"thread_ids": [thread_id]})

    assert r1.status_code == r2.status_code == 200
    assert calls["n"] == 1  # second request served from cache
    assert r2.json()["usage"][thread_id]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_langfuse_disabled_returns_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    """When fetch_usage yields {} (Langfuse off), endpoint is 200 with null totals."""
    thread_id = "usage-no-langfuse"
    cp = MemorySaver()
    model = MockChatModel(responses=[AIMessage(content="Hello!")])
    graph = create_agent(model, checkpointer=cp)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    async def _empty(session_ids):
        return {}

    monkeypatch.setattr(routes_mod, "fetch_usage", _empty)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.post("/db/threads/usage", json={"thread_ids": [thread_id]})

    assert resp.status_code == 200
    entry = resp.json()["usage"][thread_id]
    assert entry["total_tokens"] is None
    assert entry["cost_usd"] is None
    assert entry["fingerprint"] is not None  # thread exists; only usage is unknown


@pytest.mark.asyncio
async def test_usage_includes_input_output_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batch carries input/output token split when Langfuse provides it."""
    thread_id = "usage-io"
    cp = MemorySaver()
    await _seed(cp, thread_id)

    async def _stub(session_ids):
        return {thread_id: UsageRow(total_tokens=200, cost_usd=0.01, input_tokens=180, output_tokens=20)}

    monkeypatch.setattr(routes_mod, "fetch_usage", _stub)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.post("/db/threads/usage", json={"thread_ids": [thread_id]})

    entry = resp.json()["usage"][thread_id]
    assert entry["input_tokens"] == 180
    assert entry["output_tokens"] == 20


@pytest.mark.asyncio
async def test_cache_endpoint_maps_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache endpoint returns the breakdown; unchanged fingerprint serves cache."""
    thread_id = "usage-cache"
    cp = MemorySaver()
    await _seed(cp, thread_id)

    calls = {"n": 0}

    async def _stub(session_ids):
        calls["n"] += 1
        return {
            thread_id: CacheRow(
                input_total=100, input_cache_read=80, input_cache_creation=20, cache_read_pct=80.0
            )
        }

    monkeypatch.setattr(routes_mod, "fetch_cache", _stub)

    app = _make_app(cp)
    with TestClient(app) as client:
        r1 = client.post("/db/threads/usage/cache", json={"thread_ids": [thread_id]})
        r2 = client.post("/db/threads/usage/cache", json={"thread_ids": [thread_id]})

    assert r1.status_code == r2.status_code == 200
    entry = r1.json()["usage"][thread_id]
    assert entry["input_total"] == 100
    assert entry["input_cache_read"] == 80
    assert entry["cache_read_pct"] == 80.0
    assert calls["n"] == 1  # second request served from fingerprint cache


@pytest.mark.asyncio
async def test_cache_endpoint_langfuse_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache endpoint is 200 with null fields when Langfuse yields nothing."""
    thread_id = "usage-cache-off"
    cp = MemorySaver()
    await _seed(cp, thread_id)

    async def _empty(session_ids):
        return {}

    monkeypatch.setattr(routes_mod, "fetch_cache", _empty)

    app = _make_app(cp)
    with TestClient(app) as client:
        resp = client.post("/db/threads/usage/cache", json={"thread_ids": [thread_id]})

    assert resp.status_code == 200
    entry = resp.json()["usage"][thread_id]
    assert entry["cache_read_pct"] is None
    assert entry["fingerprint"] is not None
