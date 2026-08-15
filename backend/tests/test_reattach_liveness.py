"""A re-attach is proof the client is alive — it must refresh the park.

`SSEReplayMiddleware` short-circuits a re-attach POST: it never reaches an agent
loop, so nothing told the parked `LiveSession` that the app had come back. A
client that backgrounded (suspending the liveness grace, per
`claude.registry.claim_call`) and then re-attached stayed marked backgrounded
until its next keepalive ping, and its last-ping clock stayed stale — the very
moment it demonstrably reconnected.

The middleware can't import a harness (that boundary is one-way), so it exposes
`register_reattach_observer` and the Claude loop registers its own hook.
"""

from __future__ import annotations

import asyncio

import pytest

from pupa_backend import sse_replay
from pupa_backend.harnesses.claude import registry


@pytest.fixture(autouse=True)
def _clean_observers():
    before = list(sse_replay._REATTACH_OBSERVERS)
    sse_replay._REATTACH_OBSERVERS.clear()
    sse_replay.reset_logs()
    yield
    sse_replay._REATTACH_OBSERVERS.clear()
    sse_replay._REATTACH_OBSERVERS.extend(before)


# --------------------------------------------------------------------------- #
# The observer hook itself
# --------------------------------------------------------------------------- #


def test_observers_are_notified_with_the_thread_id() -> None:
    seen: list[str] = []
    sse_replay.register_reattach_observer(seen.append)

    sse_replay.notify_reattach("thread-a")

    assert seen == ["thread-a"]


def test_a_failing_observer_cannot_break_the_reattach() -> None:
    """The replay tail is the point of a re-attach; a hook must never cost it."""
    seen: list[str] = []

    def _boom(_thread_id: str) -> None:
        raise RuntimeError("observer exploded")

    sse_replay.register_reattach_observer(_boom)
    sse_replay.register_reattach_observer(seen.append)

    sse_replay.notify_reattach("thread-b")  # must not raise

    assert seen == ["thread-b"]


def test_registering_the_same_observer_twice_notifies_once() -> None:
    """Harness registration runs per mounted path (`/` and `/harnesses/…`)."""
    seen: list[str] = []
    sse_replay.register_reattach_observer(seen.append)
    sse_replay.register_reattach_observer(seen.append)

    sse_replay.notify_reattach("thread-c")

    assert seen == ["thread-c"]


# --------------------------------------------------------------------------- #
# The Claude loop's hook: a re-attach clears `backgrounded` and re-arms the grace
# --------------------------------------------------------------------------- #


async def test_reattach_marks_a_parked_session_foregrounded() -> None:
    session = registry.create("t-reattach-live")
    await session.keepalive(backgrounded=True)
    assert session.client_backgrounded is True
    stale_ping = session.last_keepalive

    registry.note_reattach("t-reattach-live")
    await asyncio.sleep(0)

    assert session.client_backgrounded is False, "re-attach left the session backgrounded"
    assert session.last_keepalive > stale_ping, "re-attach did not re-arm the grace"

    await registry.remove("t-reattach-live")


async def test_reattach_extends_a_parked_handler_deadline() -> None:
    """The observable effect: a handler about to fail the liveness grace survives
    because the app demonstrably came back."""
    session = registry.create("t-reattach-grace")
    await session.register_pending("call-1", "lsMemories", {})
    await session.keepalive()  # foreground ping → grace-bound from here

    claimed: asyncio.Future = asyncio.get_running_loop().create_future()

    async def _handler() -> None:
        try:
            claimed.set_result(await session.claim_call("lsMemories", {}, timeout=5.0))
        except asyncio.TimeoutError as exc:  # pragma: no cover — the failure mode
            claimed.set_exception(exc)

    handler = asyncio.ensure_future(_handler())
    await asyncio.sleep(0)

    # Push the last ping far enough back that the grace has already lapsed.
    session.last_keepalive -= registry.liveness_grace() * 2
    registry.note_reattach("t-reattach-grace")
    await asyncio.sleep(0)

    await session.resolve_results([{"toolCallId": "call-1", "content": "ok"}])
    result = await asyncio.wait_for(claimed, timeout=1.0)

    assert "ok" in str(result)
    await handler
    await registry.remove("t-reattach-grace")


def test_note_reattach_on_an_unknown_thread_is_a_noop() -> None:
    registry.note_reattach("t-reattach-absent")  # must not raise


async def test_note_reattach_on_a_disposed_session_is_a_noop() -> None:
    session = registry.create("t-reattach-disposed")
    await session.dispose()
    before = session.last_keepalive

    registry.note_reattach("t-reattach-disposed")
    await asyncio.sleep(0)

    assert session.last_keepalive == before


# --------------------------------------------------------------------------- #
# End to end: a re-attach POST through the middleware fires the hook
# --------------------------------------------------------------------------- #


async def test_reattach_post_notifies_observers() -> None:
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    seen: list[str] = []
    sse_replay.register_reattach_observer(seen.append)

    app = FastAPI()
    app.add_middleware(sse_replay.SSEReplayMiddleware)

    @app.post("/")
    async def _run():  # noqa: ANN202 — test route
        async def _gen():
            yield "data: {}\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # A normal run first, so the thread has a replay log to re-attach to.
        await client.post("/", json={"threadId": "t-e2e", "forwardedProps": {}})
        assert seen == [], "a normal run must not count as a re-attach"

        await client.post(
            "/",
            json={
                "threadId": "t-e2e",
                "forwardedProps": {"command": {"reattach": {"after_seq": -1}}},
            },
        )

    assert seen == ["t-e2e"]
