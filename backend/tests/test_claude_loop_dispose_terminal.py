"""Teardown must end the attached SSE, not strand it.

`LiveSession.dispose()` runs when a thread's session is replaced (a fresh
new-turn POST reusing the thread) or evicted by the idle sweeper. It used to
cancel the pump and disconnect the SDK client without putting anything on the
queue, so an `attach()` drain sat blocked on `queue.get()` forever: no
`RunFinished`, no `RunError`, and a client that re-attached later got an empty
`live=0` replay tail — the chat silently stopped.

Second hazard: `attach()` removes the session on a terminal sentinel. When the
stale session's drain wakes up *after* a newer session has claimed the thread,
that removal must not evict the newcomer.
"""

from __future__ import annotations

import asyncio

import pytest
from ag_ui.core.events import RunErrorEvent

from pupa_backend.harnesses.claude import events as cl_events
from pupa_backend.harnesses.claude import registry


async def _drain(session: registry.LiveSession) -> list:
    return [event async for event in registry.attach(session)]


async def test_dispose_emits_terminal_error_and_ends_attach() -> None:
    """A parked session torn down mid-turn ends its SSE with a RunError."""
    session = registry.create("t-dispose-terminal")
    drain = asyncio.ensure_future(_drain(session))
    await asyncio.sleep(0)  # let the drain block on the queue

    await session.dispose()

    events = await asyncio.wait_for(drain, timeout=1.0)
    assert events, "dispose() stranded the attached SSE with no terminal event"
    assert isinstance(events[-1], RunErrorEvent)


async def test_replaced_session_teardown_keeps_the_new_session() -> None:
    """`create()` disposing the stale session must not evict its replacement."""
    stale = registry.create("t-dispose-replace")
    drain = asyncio.ensure_future(_drain(stale))
    await asyncio.sleep(0)

    fresh = registry.create("t-dispose-replace")  # fire-and-forget disposes `stale`
    await asyncio.wait_for(drain, timeout=1.0)

    assert registry.get("t-dispose-replace") is fresh
    await registry.remove("t-dispose-replace")


async def test_dispose_is_idempotent() -> None:
    """A second dispose (e.g. `attach()` → `remove()` after the first) is a no-op."""
    session = registry.LiveSession(thread_id="t-dispose-twice")
    await session.dispose()
    drained: list = []
    while not session.queue.empty():
        drained.append(session.queue.get_nowait())
    await session.dispose()
    assert session.queue.empty(), "second dispose re-queued terminal events"
    assert drained, "first dispose queued nothing"


async def test_dispose_denies_a_parked_permission_with_an_error_event() -> None:
    """A parked approval prompt torn down must not leave the user staring at a
    question nobody will ever answer."""
    session = registry.create("t-dispose-permission")
    loop = asyncio.get_running_loop()
    session.pending_decision = loop.create_future()
    drain = asyncio.ensure_future(_drain(session))
    await asyncio.sleep(0)

    await session.dispose()

    events = await asyncio.wait_for(drain, timeout=1.0)
    assert session.pending_decision.result() is False
    assert isinstance(events[-1], RunErrorEvent)


@pytest.mark.parametrize("terminal", ["finish", "error"])
async def test_normal_terminal_still_wins(terminal: str) -> None:
    """A pump that finished normally keeps its own terminal — the removal-time
    dispose must not turn a clean finish into an error."""
    session = registry.create(f"t-terminal-{terminal}")
    if terminal == "finish":
        session.mark_finish()
    else:
        session.emit(cl_events.run_error("boom"))
        session.mark_error()

    events = await asyncio.wait_for(_drain(session), timeout=1.0)
    if terminal == "finish":
        assert events == []
    else:
        assert isinstance(events[-1], RunErrorEvent)
        assert "boom" in events[-1].message
    assert registry.get(f"t-terminal-{terminal}") is None
