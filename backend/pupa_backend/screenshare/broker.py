"""In-memory pairing state for screen-share sessions.

A `ShareSession` holds the publisher and (optional) viewer WebSocket for a
given `share_id`. Mutations are serialised by an asyncio lock so the
publisher/viewer registration races resolve deterministically.

v1 enforces single-viewer: a second viewer for an already-paired session is
rejected with WS close code 4409. State shape (`viewer: WebSocket | None`)
can grow to a list later without a protocol change.
"""

import asyncio
from dataclasses import dataclass

from fastapi import WebSocket


@dataclass
class ShareSession:
    share_id: str
    publisher: WebSocket | None = None
    viewer: WebSocket | None = None


class Broker:
    def __init__(self) -> None:
        self._sessions: dict[str, ShareSession] = {}
        self._lock = asyncio.Lock()

    async def register_publisher(
        self, share_id: str, ws: WebSocket
    ) -> ShareSession | None:
        async with self._lock:
            existing = self._sessions.get(share_id)
            if existing and existing.publisher is not None:
                return None
            session = existing or ShareSession(share_id=share_id)
            session.publisher = ws
            self._sessions[share_id] = session
            return session

    async def register_viewer(
        self, share_id: str, ws: WebSocket
    ) -> tuple[ShareSession | None, int | None]:
        async with self._lock:
            session = self._sessions.get(share_id)
            if session is None or session.publisher is None:
                return None, 4404
            if session.viewer is not None:
                return None, 4409
            session.viewer = ws
            return session, None

    async def remove_publisher(self, share_id: str) -> ShareSession | None:
        async with self._lock:
            session = self._sessions.get(share_id)
            if session is None:
                return None
            session.publisher = None
            viewer_to_notify = session.viewer
            if session.viewer is None:
                self._sessions.pop(share_id, None)
            return ShareSession(share_id=share_id, viewer=viewer_to_notify)

    async def remove_viewer(self, share_id: str) -> ShareSession | None:
        async with self._lock:
            session = self._sessions.get(share_id)
            if session is None:
                return None
            publisher_to_notify = session.publisher
            session.viewer = None
            if session.publisher is None:
                self._sessions.pop(share_id, None)
            return ShareSession(share_id=share_id, publisher=publisher_to_notify)

    async def sole_publisher_share_id(self) -> str | None:
        """Return the share_id of the only active publisher, or None if 0 or 2+."""
        async with self._lock:
            active = [sid for sid, s in self._sessions.items() if s.publisher is not None]
            return active[0] if len(active) == 1 else None


broker = Broker()
