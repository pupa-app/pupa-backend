"""Single-consumer handoff for live harness event queues."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

_HANDOVER_TIMEOUT = 2.0


@dataclass
class Attachment:
    """One event-stream consumer's lease on a live session queue."""

    stop: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)


async def single_consumer_events(
    session: Any,
    *,
    logger: logging.Logger,
    label: str,
) -> AsyncIterator[Any]:
    """Drain ``session.queue``, displacing any older consumer without loss.

    The session supplies ``attachment``, ``pushback``, ``thread_id``, and
    ``touch``. Sentinels are deliberately yielded so each harness retains its
    own terminal semantics.
    """
    previous = session.attachment
    mine = Attachment()
    session.attachment = mine
    if previous is not None:
        previous.stop.set()
        try:
            await asyncio.wait_for(previous.done.wait(), timeout=_HANDOVER_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info(
                "%s: stream handover timed out thread_id=%s; continuing",
                label,
                session.thread_id,
            )
    session.touch()
    try:
        while True:
            if session.pushback:
                item = session.pushback.pop(0)
            else:
                get = asyncio.create_task(session.queue.get())
                displaced = asyncio.create_task(mine.stop.wait())
                done, _pending = await asyncio.wait(
                    {get, displaced}, return_when=asyncio.FIRST_COMPLETED
                )
                if displaced in done:
                    if get in done:
                        session.pushback.insert(0, get.result())
                    else:
                        get.cancel()
                    return
                displaced.cancel()
                item = get.result()
            yield item
            session.touch()
    finally:
        mine.done.set()
        if session.attachment is mine:
            session.attachment = None
