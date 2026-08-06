"""Transport-level SSE keep-alive.

The iOS client speaks to `POST /` over a single Server-Sent Events stream and
uses `URLSession`, whose per-request idle timeout (`timeoutIntervalForRequest`,
~60s by default) only resets when bytes arrive. A turn that emits no AG-UI
events for >60s — a long Claude think/tool turn, or a slow LangGraph tool call —
trips `NSURLErrorDomain -1001 "request timed out"` even though the server is
healthy and still working.

Idle-timeout is a transport concern, not an agent-loop concern, so the fix lives
here at the app boundary rather than inside any one loop's drain. This middleware
wraps every `text/event-stream` response body and, whenever the upstream
generator is silent for `_INTERVAL` seconds, emits an SSE comment line
(`: keep-alive\\n\\n`). Comments are ignored by every SSE parser but count as
bytes, so the client's idle timer resets and a long silent turn survives.

Both `POST /` handlers (the Claude Code loop and the `ag_ui_langgraph` LangGraph
endpoint) and any future SSE route inherit this for free — no per-route opt-in.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse

_SSE_MEDIA_TYPE = "text/event-stream"

# SSE comment line. The leading ':' marks a comment that the client's event
# parser discards, but it still arrives as bytes and resets the idle timer.
_KEEP_ALIVE = b": keep-alive\n\n"

# Returned by `_safe_anext` when the upstream generator is exhausted, so the
# pull task carries a sentinel rather than raising `StopAsyncIteration` (which
# can't cross a Future boundary cleanly).
_STREAM_DONE = object()


async def _safe_anext(iterator):
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return _STREAM_DONE


def _interval() -> float:
    """Max gap between bytes on an SSE stream, in seconds.

    Default 15s — comfortably under the client's ~60s idle timeout with room for
    a missed beat. `PUPA_SSE_KEEPALIVE_INTERVAL <= 0` disables the keep-alive.
    """
    raw = os.getenv("PUPA_SSE_KEEPALIVE_INTERVAL")
    if raw is None:
        return 15.0
    try:
        return float(raw)
    except ValueError:
        return 15.0


def _is_sse(response) -> bool:
    if getattr(response, "media_type", None) == _SSE_MEDIA_TYPE:
        return True
    # `media_type` can be unset when a handler builds the response from raw
    # headers; fall back to the content-type header (may carry a charset).
    content_type = response.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip() == _SSE_MEDIA_TYPE


class SSEKeepAliveMiddleware(BaseHTTPMiddleware):
    """Inject SSE keep-alive comments into idle `text/event-stream` responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        interval = _interval()
        if interval <= 0 or not _is_sse(response):
            return response  # disabled, or a plain JSON/REST response — pass through

        upstream = response.body_iterator

        async def _keep_alive():
            # Pull the next chunk in its own task and race the heartbeat timer
            # against it with `asyncio.wait` (which does NOT cancel the task on
            # timeout). Cancelling a generator's `__anext__()` mid-`await` would
            # raise CancelledError *inside* the generator and kill it, dropping
            # the chunk — so the pull task is left running across heartbeats and
            # only ever cancelled when the client goes away (the `finally`).
            pull: asyncio.Task | None = None
            try:
                while True:
                    if pull is None:
                        pull = asyncio.ensure_future(_safe_anext(upstream))
                    done, _ = await asyncio.wait({pull}, timeout=interval)
                    if not done:
                        yield _KEEP_ALIVE  # upstream still silent — keep bytes flowing
                        continue
                    chunk = pull.result()  # re-raises a real upstream error
                    pull = None
                    if chunk is _STREAM_DONE:
                        break
                    yield chunk
            finally:
                # Client disconnect / shutdown closes this generator; drop the
                # in-flight pull and the upstream so neither leaks.
                if pull is not None and not pull.done():
                    pull.cancel()
                    # Await the cancelled pull so the upstream generator's
                    # in-flight `__anext__()` unwinds before we close it.
                    # Skipping this races `aclose()` against a still-running
                    # generator -> "aclose(): asynchronous generator is already
                    # running".
                    with contextlib.suppress(BaseException):
                        await pull
                aclose = getattr(upstream, "aclose", None)
                if aclose is not None:
                    await aclose()

        wrapped = StreamingResponse(
            _keep_alive(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type or _SSE_MEDIA_TYPE,
            background=response.background,
        )
        return wrapped
