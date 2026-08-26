"""Tests for the transport-level resumable-SSE replay middleware.

Driven through a tiny FastAPI app over `httpx.ASGITransport` (same style as
`test_sse_keepalive.py`) so the assertions exercise the real Starlette
response path. The fake `POST /` handler stands in for either agent loop —
the middleware must not care which one produced the stream.
"""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import pupa_backend.sse_replay as sse_replay
from pupa_backend.sse_replay import (
    HEADER_LIVE,
    HEADER_NEXT_SEQ,
    HEADER_OLDEST_SEQ,
    FrameSplitter,
    SSEReplayMiddleware,
    _extract,
)


@pytest.fixture(autouse=True)
def _clean_logs():
    sse_replay.reset_logs()
    yield
    sse_replay.reset_logs()


def _app(monkeypatch: pytest.MonkeyPatch, ttl: str = "60") -> FastAPI:
    monkeypatch.setenv("PUPA_SSE_REPLAY_TTL", ttl)
    app = FastAPI()
    app.add_middleware(SSEReplayMiddleware)
    return app


def _run_body(thread_id: str = "t1") -> dict:
    return {"threadId": thread_id, "runId": "r1", "messages": [], "tools": []}


def _reattach_body(thread_id: str = "t1", after_seq: int = -1) -> dict:
    return {
        "threadId": thread_id,
        "runId": "r2",
        "forwardedProps": {"command": {"reattach": {"after_seq": after_seq}}},
    }


async def _post(app: FastAPI, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/", json=body)


def _events(sse_body: bytes) -> list[tuple[int | None, str]]:
    """Parse an SSE byte body into [(id, data), ...], ignoring comments."""
    out: list[tuple[int | None, str]] = []
    for raw_frame in sse_body.split(b"\n\n"):
        frame_id: int | None = None
        data: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith(b":") or not line:
                continue
            if line.startswith(b"id:"):
                frame_id = int(line[3:].strip())
            elif line.startswith(b"data:"):
                data.append(line[5:].strip().decode())
        if data:
            out.append((frame_id, "\n".join(data)))
    return out


# ---------------------------------------------------------------------------
# FrameSplitter / _extract units
# ---------------------------------------------------------------------------


def test_frame_splitter_handles_partial_and_multi_frames() -> None:
    s = FrameSplitter()
    assert s.feed(b"data: a\n\ndata: b") == [b"data: a"]
    assert s.feed(b"\n\nda") == [b"data: b"]
    assert s.feed(b"ta: c\n\n") == [b"data: c"]
    assert s.flush() is None


def test_frame_splitter_crlf_and_trailing_flush() -> None:
    s = FrameSplitter()
    assert s.feed(b"data: a\r\n\r\ndata: tail") == [b"data: a"]
    assert s.flush() == b"data: tail"


def test_extract_run_and_reattach() -> None:
    assert _extract(json.dumps(_run_body("th")).encode()) == ("th", None)
    tid, seq = _extract(json.dumps(_reattach_body("th", 7)).encode())
    assert (tid, seq) == ("th", 7)
    assert _extract(b"not json") == (None, None)
    assert _extract(json.dumps({"noThread": True}).encode()) == (None, None)


# ---------------------------------------------------------------------------
# End-to-end middleware behavior
# ---------------------------------------------------------------------------


async def test_run_stream_is_sequenced_and_passes_through(monkeypatch) -> None:
    app = _app(monkeypatch)

    @app.post("/")
    async def run():
        async def gen():
            yield b"data: one\n\n"
            yield b"data: two\n\ndata: three\n\n"  # two frames in one chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    r = await _post(app, _run_body())
    assert r.status_code == 200
    assert _events(r.content) == [(0, "one"), (1, "two"), (2, "three")]
    assert r.headers[HEADER_LIVE] == "1"  # pump had not finished at header time


async def test_reattach_replays_full_turn_after_finish(monkeypatch) -> None:
    """Catch-up-after-kill: the retained log IS the completed-turn snapshot."""
    app = _app(monkeypatch)

    @app.post("/")
    async def run():
        async def gen():
            yield b"data: hello\n\n"
            yield b"data: world\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    await _post(app, _run_body())
    r = await _post(app, _reattach_body(after_seq=-1))
    assert r.status_code == 200
    assert _events(r.content) == [(0, "hello"), (1, "world")]
    assert r.headers[HEADER_LIVE] == "0"
    assert r.headers[HEADER_NEXT_SEQ] == "2"
    assert r.headers[HEADER_OLDEST_SEQ] == "0"


async def test_reattach_after_seq_skips_already_seen(monkeypatch) -> None:
    app = _app(monkeypatch)

    @app.post("/")
    async def run():
        async def gen():
            for i in range(4):
                yield f"data: e{i}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    await _post(app, _run_body())
    r = await _post(app, _reattach_body(after_seq=1))
    assert _events(r.content) == [(2, "e2"), (3, "e3")]


async def test_reattach_unknown_thread_is_204(monkeypatch) -> None:
    app = _app(monkeypatch)
    r = await _post(app, _reattach_body("nope"))
    assert r.status_code == 204


async def _drive_until_disconnect(app, body: bytes, produced: list) -> None:
    """Run one POST through the raw ASGI interface and hang up mid-stream.

    `httpx.ASGITransport` buffers whole responses, so it cannot express this;
    a real disconnect is the whole point here.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},  # what uvicorn sends
        "http_version": "1.1",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    sent_body = False
    disconnected = asyncio.Event()

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    chunks = 0

    async def send(message):
        nonlocal chunks
        if message["type"] == "http.response.body" and message.get("body"):
            chunks += 1
            if chunks == 1:
                # The client is gone from here on.
                disconnected.set()

    await app(scope, receive, send)


async def test_run_survives_a_real_client_disconnect(monkeypatch) -> None:
    """The promise in this module's docstring, exercised over a real ASGI
    disconnect rather than at the generator level.

    A client that dies mid-turn must not take the run with it: the agent keeps
    producing, the pump keeps logging, and the re-attach that follows the
    client's relaunch serves what it missed. Without that the turn is lost —
    the client reattaches, finds nothing past its cursor, and reports the turn
    settled with the work gone.
    """
    app = _app(monkeypatch)
    produced: list[int] = []
    finished = asyncio.Event()

    @app.post("/")
    async def run() -> StreamingResponse:
        async def body():
            for i in range(6):
                produced.append(i)
                yield f"id: {i}\ndata: frame-{i}\n\n".encode()
                await asyncio.sleep(0.01)
            finished.set()

        return StreamingResponse(body(), media_type="text/event-stream")

    import json

    await _drive_until_disconnect(app, json.dumps(_run_body()).encode(), produced)

    # The run must reach its end even though nobody was listening.
    await asyncio.wait_for(finished.wait(), timeout=2)
    assert produced == list(range(6)), f"the run stopped when the client did: {produced}"

    # And what it produced must be reattachable.
    resp = await _post(app, _reattach_body(after_seq=-1))
    assert resp.status_code == 200, "the thread was evicted with the socket"
    seqs = [seq for seq, _ in _events(resp.content)]
    assert len(seqs) == 6, f"the tail lost frames produced after the disconnect: {seqs}"


async def test_tail_close_does_not_kill_pump_or_other_tails() -> None:
    """The core promise, exercised at the generator level (httpx's
    ASGITransport buffers whole responses, so it cannot simulate a real
    mid-stream disconnect): closing one tail — as Starlette does when the
    client's socket dies — must not stall `append` (the pump) and a later
    tail must still see everything, including frames logged *after* the
    disconnect."""
    log = sse_replay.ReplayLog("t", max_events=100, max_bytes=10_000)
    log.mark_pump_started()
    log.append(b"data: early")

    dropped = log.tail(-1)
    assert (await dropped.__anext__()) == b"id: 0\ndata: early\n\n"
    # Park the tail in its wait-for-more state, then kill it (disconnect).
    parked = asyncio.ensure_future(dropped.__anext__())
    await asyncio.sleep(0.01)
    parked.cancel()
    await asyncio.gather(parked, return_exceptions=True)
    await dropped.aclose()

    # Pump continues unaffected.
    log.append(b"data: late")
    log.mark_pump_done()

    replayed = [frame async for frame in log.tail(-1)]
    assert replayed == [b"id: 0\ndata: early\n\n", b"id: 1\ndata: late\n\n"]


async def test_reattach_live_tails_until_run_ends(monkeypatch) -> None:
    """A re-attach issued while the pump is still running must replay what is
    buffered and then stream the rest live."""
    app = _app(monkeypatch)
    release = asyncio.Event()
    started = asyncio.Event()

    @app.post("/")
    async def run():
        async def gen():
            yield b"data: first\n\n"
            started.set()
            await release.wait()
            yield b"data: second\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_task = asyncio.create_task(client.post("/", json=_run_body()))
        await asyncio.wait_for(started.wait(), timeout=2.0)  # pump is mid-run

        reattach = asyncio.create_task(client.post("/", json=_reattach_body(after_seq=-1)))
        await asyncio.sleep(0.05)  # reattach begins while the run is live
        release.set()
        r = await asyncio.wait_for(reattach, timeout=2.0)
        await asyncio.wait_for(run_task, timeout=2.0)

    assert _events(r.content) == [(0, "first"), (1, "second")]
    assert r.headers[HEADER_LIVE] == "1"  # was live at reattach time


async def test_seq_continues_across_rounds_on_same_thread(monkeypatch) -> None:
    """Interrupt/resume rounds are separate POSTs on one thread — the log's
    seq must stay monotonic across them so client cursors never rewind."""
    app = _app(monkeypatch)
    round_no = {"n": 0}

    @app.post("/")
    async def run():
        round_no["n"] += 1
        n = round_no["n"]

        async def gen():
            yield f"data: round{n}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    await _post(app, _run_body())
    await _post(app, _run_body())
    r = await _post(app, _reattach_body(after_seq=-1))
    assert _events(r.content) == [(0, "round1"), (1, "round2")]


async def test_ring_caps_evict_oldest(monkeypatch) -> None:
    monkeypatch.setenv("PUPA_SSE_REPLAY_MAX_EVENTS", "2")
    app = _app(monkeypatch)

    @app.post("/")
    async def run():
        async def gen():
            for i in range(5):
                yield f"data: e{i}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    await _post(app, _run_body())
    r = await _post(app, _reattach_body(after_seq=-1))
    assert _events(r.content) == [(3, "e3"), (4, "e4")]
    assert r.headers[HEADER_OLDEST_SEQ] == "3"


async def test_ttl_sweep_evicts_idle_logs(monkeypatch) -> None:
    app = _app(monkeypatch, ttl="0.01")

    @app.post("/")
    async def run():
        async def gen():
            yield b"data: x\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    await _post(app, _run_body())
    await asyncio.sleep(0.05)
    r = await _post(app, _reattach_body(after_seq=-1))
    assert r.status_code == 204  # swept


async def test_disabled_by_ttl_zero_passes_through(monkeypatch) -> None:
    app = _app(monkeypatch, ttl="0")

    @app.post("/")
    async def run():
        async def gen():
            yield b"data: raw\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    r = await _post(app, _run_body())
    assert b"id:" not in r.content  # untouched stream, no sequencing
    assert HEADER_LIVE not in r.headers


async def test_non_sse_and_other_routes_untouched(monkeypatch) -> None:
    app = _app(monkeypatch)

    @app.post("/")
    async def run():
        return JSONResponse({"ok": True})

    @app.get("/health")
    async def health():
        return JSONResponse({"up": True})

    r = await _post(app, _run_body())
    assert r.json() == {"ok": True}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r2 = await client.get("/health")
    assert r2.json() == {"up": True}


async def test_comment_frames_not_logged(monkeypatch) -> None:
    """Keep-alive style comments must never occupy replay-log slots."""
    app = _app(monkeypatch)

    @app.post("/")
    async def run():
        async def gen():
            yield b": keep-alive\n\n"
            yield b"data: real\n\n"
            yield b": keep-alive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    await _post(app, _run_body())
    r = await _post(app, _reattach_body(after_seq=-1))
    assert _events(r.content) == [(0, "real")]
