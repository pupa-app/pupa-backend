"""Tests for the transport-level SSE keep-alive middleware.

The middleware is driven through a tiny FastAPI app over `httpx.ASGITransport`
so the assertions exercise the real Starlette response path, not a hand-rolled
stand-in. A low `PUPA_SSE_KEEPALIVE_INTERVAL` keeps the timing tests fast.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from pupa_backend.sse_keepalive import SSEKeepAliveMiddleware, _is_sse


def _app(interval: str, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("PUPA_SSE_KEEPALIVE_INTERVAL", interval)
    app = FastAPI()
    app.add_middleware(SSEKeepAliveMiddleware)
    return app


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_heartbeat_emitted_while_upstream_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app("0.05", monkeypatch)

    @app.get("/sse")
    async def sse():
        async def gen():
            await asyncio.sleep(0.2)  # > interval — should provoke heartbeats
            yield b"data: real\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    r = await _get(app, "/sse")
    body = r.content
    # Keep-alive comment(s) emitted, and emitted *before* the real event.
    assert b": keep-alive\n\n" in body
    assert body.index(b": keep-alive") < body.index(b"data: real")
    # The real upstream event is not dropped by the heartbeat race.
    assert b"data: real\n\n" in body


async def test_no_heartbeat_when_events_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app("5.0", monkeypatch)  # interval well above the stream's lifetime

    @app.get("/sse")
    async def sse():
        async def gen():
            yield b"data: a\n\n"
            yield b"data: b\n\n"
            yield b"data: c\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    r = await _get(app, "/sse")
    # No heartbeat interleaved; all chunks delivered in order, stream ends clean.
    assert b": keep-alive" not in r.content
    assert r.content == b"data: a\n\ndata: b\n\ndata: c\n\n"


async def test_non_sse_response_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app("0.05", monkeypatch)

    @app.get("/json")
    async def json_route():
        return JSONResponse({"ok": True})

    r = await _get(app, "/json")
    assert r.json() == {"ok": True}
    assert r.headers["content-type"].startswith("application/json")
    assert b": keep-alive" not in r.content


async def test_disabled_when_interval_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app("0", monkeypatch)

    @app.get("/sse")
    async def sse():
        async def gen():
            await asyncio.sleep(0.1)
            yield b"data: real\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    r = await _get(app, "/sse")
    assert b": keep-alive" not in r.content
    assert b"data: real\n\n" in r.content


async def test_client_disconnect_while_pull_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: closing the wrapped stream while the upstream generator's
    # `__anext__()` is still running must NOT raise "aclose(): asynchronous
    # generator is already running". Reproduces an iOS app-close disconnect
    # mid-turn, where Starlette calls `aclose()` on the keep-alive generator
    # while the in-flight pull task is parked inside upstream.__anext__().
    monkeypatch.setenv("PUPA_SSE_KEEPALIVE_INTERVAL", "0.05")

    upstream_closed = asyncio.Event()

    async def upstream():
        try:
            yield b"data: first\n\n"
            await asyncio.sleep(3600)  # park inside __anext__ forever
            yield b"data: never\n\n"
        finally:
            upstream_closed.set()

    async def call_next(_request: Request) -> StreamingResponse:
        return StreamingResponse(upstream(), media_type="text/event-stream")

    middleware = SSEKeepAliveMiddleware(app=None)
    request = Request({"type": "http", "headers": []})
    response = await middleware.dispatch(request, call_next)

    wrapped = response.body_iterator
    assert await wrapped.__anext__() == b"data: first\n\n"
    # Next pull parks in the 3600s sleep; interval fires first -> a heartbeat,
    # leaving the pull task in-flight exactly as at a real disconnect.
    assert await wrapped.__anext__() == b": keep-alive\n\n"

    # Client goes away: Starlette closes the wrapped generator. Must not raise.
    await wrapped.aclose()
    # Upstream generator was closed too (no leak).
    assert upstream_closed.is_set()


def test_is_sse_detection() -> None:
    sse = StreamingResponse(iter(()), media_type="text/event-stream")
    assert _is_sse(sse)
    json_resp = JSONResponse({"a": 1})
    assert not _is_sse(json_resp)
