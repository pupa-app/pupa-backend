"""One session queue, one consumer.

A resume POST can arrive while the previous request's `attach()` generator is
still draining — the pump survives a client disconnect by design, and the client
re-POSTs a resume whose response was lost. Two generators on one
`asyncio.Queue` each take a *share* of the events: the turn is split across two
SSE responses, and both append into the same replay log through independent task
chains, so the log's frame order is no longer the turn's order.

`attach()` therefore displaces the consumer already on the queue.
"""

from __future__ import annotations

import asyncio

import pytest

from pupa_backend.harnesses.claude import registry


async def _drain(session: registry.LiveSession, out: list) -> None:
    async for event in registry.attach(session):
        out.append(event)


@pytest.mark.asyncio
async def test_second_attach_takes_every_event() -> None:
    session = registry.LiveSession(thread_id="t-single")
    first: list = []
    second: list = []

    first_task = asyncio.create_task(_drain(session, first))
    await asyncio.sleep(0)  # let it park on queue.get()

    second_task = asyncio.create_task(_drain(session, second))
    await asyncio.sleep(0)

    for i in range(4):
        session.emit(f"e{i}")
    session.mark_finish()

    await asyncio.wait_for(second_task, timeout=2)
    await asyncio.wait_for(first_task, timeout=2)

    assert second == ["e0", "e1", "e2", "e3"], "the live request must see the whole turn"
    assert first == [], "the displaced consumer must stop taking events"


@pytest.mark.asyncio
async def test_displaced_consumer_does_not_swallow_an_event() -> None:
    """The displacement must not eat an event already handed to the old
    generator: whatever it had in flight belongs to the new one."""
    session = registry.LiveSession(thread_id="t-handoff")
    first: list = []
    second: list = []

    first_task = asyncio.create_task(_drain(session, first))
    await asyncio.sleep(0)
    session.emit("early")            # racing the handover
    second_task = asyncio.create_task(_drain(session, second))

    session.emit("late")
    session.mark_finish()

    await asyncio.wait_for(second_task, timeout=2)
    await asyncio.wait_for(first_task, timeout=2)

    assert first == [], "the displaced consumer must hand its in-flight event back"
    assert second[:2] == ["early", "late"], "no event may be dropped or duplicated"


@pytest.mark.asyncio
async def test_single_attach_is_unchanged() -> None:
    session = registry.LiveSession(thread_id="t-solo")
    out: list = []
    task = asyncio.create_task(_drain(session, out))

    session.emit("a")
    session.emit("b")
    session.mark_interrupt()

    await asyncio.wait_for(task, timeout=2)
    assert out == ["a", "b"]
