"""Client liveness heartbeat for parked frontend tools.

While a frontend tool is parked the SSE is closed — no socket detects a dead
app. The client now POSTs `command.keepalive {state}` every ~10s while a tool
is in flight; `claim_call` waits on a liveness deadline (`last_keepalive +
grace`) instead of burning the full per-tool wall, which stays as the absolute
cap. An explicit `state: "background"` ping falls back to the absolute cap
(Option B: a subagent survives a brief background; a dead app stays bounded).
Clients that never ping keep the old full-wall behaviour.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from pupa_backend.harnesses.claude import registry


# --------------------------------------------------------------------------- #
# claim_call liveness deadline (registry level)
# --------------------------------------------------------------------------- #

async def test_lost_liveness_fails_fast_before_wall(monkeypatch: pytest.MonkeyPatch) -> None:
    """One ping then silence (no background notice) → handler fails ~grace after
    the last ping, long before the absolute wall."""
    monkeypatch.setenv("PUPA_FRONTEND_LIVENESS_GRACE", "0.2")
    loop = asyncio.get_event_loop()
    session = registry.LiveSession(thread_id="t-liveness-lost")
    await session.register_pending("c1", "invoke_agent", {"p": "x"})
    await session.keepalive()

    start = loop.time()
    with pytest.raises(asyncio.TimeoutError) as exc:
        await session.claim_call("invoke_agent", {"p": "x"}, timeout=5.0)
    elapsed = loop.time() - start

    assert elapsed < 1.0, f"expected ~grace fail, burned {elapsed:.2f}s"
    assert "liveness" in str(exc.value)


async def test_pings_keep_handler_alive_past_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuous pings extend the liveness deadline across many grace windows;
    the handler survives until the resume delivers the result."""
    monkeypatch.setenv("PUPA_FRONTEND_LIVENESS_GRACE", "0.15")
    session = registry.LiveSession(thread_id="t-liveness-alive")
    await session.register_pending("c1", "invoke_agent", {"p": "x"})
    await session.keepalive()

    async def _ping_then_resolve() -> None:
        for _ in range(8):  # 0.4s of pings ≫ one 0.15s grace window
            await asyncio.sleep(0.05)
            await session.keepalive()
        await session.resolve_results([{"toolCallId": "c1", "content": "done"}])

    pinger = asyncio.ensure_future(_ping_then_resolve())
    result = await asyncio.wait_for(
        session.claim_call("invoke_agent", {"p": "x"}, timeout=5.0), timeout=2.0
    )
    await pinger
    assert result == {"content": [{"type": "text", "text": "done"}]}


async def test_never_pinged_keeps_full_wall_compat() -> None:
    """A client that never pings (older app) keeps the pre-heartbeat behaviour: the
    handler waits the full per-tool wall."""
    loop = asyncio.get_event_loop()
    session = registry.LiveSession(thread_id="t-no-pings")
    await session.register_pending("c1", "addTrackerItems", {"x": 1})

    start = loop.time()
    with pytest.raises(asyncio.TimeoutError):
        await session.claim_call("addTrackerItems", {"x": 1}, timeout=0.4)
    elapsed = loop.time() - start
    assert elapsed >= 0.4, f"never-pinged client must keep the wall, got {elapsed:.3f}s"


async def test_background_ping_falls_back_to_absolute_wall(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit background notice suspends the liveness deadline (iOS freezes
    timers in background) — the absolute per-tool wall still bounds the park."""
    monkeypatch.setenv("PUPA_FRONTEND_LIVENESS_GRACE", "0.1")
    loop = asyncio.get_event_loop()
    session = registry.LiveSession(thread_id="t-backgrounded")
    await session.register_pending("c1", "invoke_agent", {"p": "x"})
    await session.keepalive()
    await session.keepalive(backgrounded=True)

    start = loop.time()
    with pytest.raises(asyncio.TimeoutError) as exc:
        await session.claim_call("invoke_agent", {"p": "x"}, timeout=0.5)
    elapsed = loop.time() - start

    assert elapsed >= 0.5, f"backgrounded client fell to liveness grace: {elapsed:.3f}s"
    assert "liveness" not in str(exc.value)


async def test_foreground_ping_restores_liveness_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returning to foreground (a plain ping) re-arms the short liveness grace."""
    monkeypatch.setenv("PUPA_FRONTEND_LIVENESS_GRACE", "0.15")
    loop = asyncio.get_event_loop()
    session = registry.LiveSession(thread_id="t-foregrounded")
    await session.register_pending("c1", "invoke_agent", {"p": "x"})
    await session.keepalive(backgrounded=True)
    await session.keepalive()  # foreground again, then silence

    start = loop.time()
    with pytest.raises(asyncio.TimeoutError):
        await session.claim_call("invoke_agent", {"p": "x"}, timeout=5.0)
    elapsed = loop.time() - start
    assert elapsed < 1.0, f"foreground ping should re-arm grace, burned {elapsed:.2f}s"


async def test_ping_wakes_parked_waiter_to_extend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ping landing while the handler is parked extends the deadline in place —
    the waiter must not fail on the stale pre-ping deadline."""
    monkeypatch.setenv("PUPA_FRONTEND_LIVENESS_GRACE", "0.2")
    session = registry.LiveSession(thread_id="t-extend")
    await session.register_pending("c1", "invoke_agent", {"p": "x"})
    await session.keepalive()

    async def _late_ping_and_resolve() -> None:
        await asyncio.sleep(0.15)  # just inside the first grace window
        await session.keepalive()
        await asyncio.sleep(0.15)  # inside the SECOND window — past the first
        await session.resolve_results([{"toolCallId": "c1", "content": "late"}])

    helper = asyncio.ensure_future(_late_ping_and_resolve())
    result = await asyncio.wait_for(
        session.claim_call("invoke_agent", {"p": "x"}, timeout=5.0), timeout=2.0
    )
    await helper
    payload = result["content"][0]["text"]
    assert payload == "late"


# --------------------------------------------------------------------------- #
# Endpoint: command.keepalive branch
# --------------------------------------------------------------------------- #

def _keepalive_body(thread_id: str, state: str = "active") -> dict:
    return {
        "thread_id": thread_id,
        "run_id": "run-ka",
        "messages": [],
        "tools": [],
        "state": {},
        "context": [],
        "forwardedProps": {"command": {"keepalive": {"state": state}}},
    }


async def test_endpoint_keepalive_touches_parked_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {})
    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    session = registry.create("ka-thread")
    assert session.last_keepalive is None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/", json=_keepalive_body("ka-thread"))
        assert r.status_code == 204
        assert session.last_keepalive is not None
        assert session.client_backgrounded is False

        r = await client.post("/", json=_keepalive_body("ka-thread", state="background"))
        assert r.status_code == 204
        assert session.client_backgrounded is True

        # Unknown thread: still a clean 204 no-op (fire-and-forget ping).
        r = await client.post("/", json=_keepalive_body("nonexistent-thread"))
        assert r.status_code == 204

    await registry.remove("ka-thread")
