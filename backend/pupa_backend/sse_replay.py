"""Transport-level resumable SSE: sequenced event replay + re-attach.

The iOS client streams agent output over one `POST /` SSE response. When iOS
backgrounds (or kills) the app the socket dies and — before this module —
every event already yielded into the response was gone: the turn errored
client-side with no way to recover it.

Like `sse_keepalive`, this is a transport concern, not an agent-loop concern:
the fix lives at the app boundary so **both** `POST /` handlers (the Claude
Code loop and the `ag_ui_langgraph` LangGraph endpoint) inherit it identically
with their code untouched.

## How it works

For every SSE-producing `POST /` run, the middleware *detaches* the handler's
body iterator from the HTTP response:

  1. A background **pump** task drains the handler's generator to completion,
     splitting the byte stream into SSE frames, stamping each with a monotonic
     per-thread sequence number (emitted as the SSE `id:` field), and appending
     it to an in-memory per-thread `ReplayLog` ring buffer.
  2. The HTTP response body is a **tail reader** over that log. Client
     disconnect closes only the tail — the pump (and therefore the agent run,
     whichever loop owns it) keeps going, events keep accumulating.

A client that lost its socket re-attaches by POSTing to `/` with:

    {"threadId": "...", "forwardedProps": {"command": {"reattach": {"after_seq": N}}}}

The middleware short-circuits that request — it never reaches either agent
loop — and answers with a fresh SSE that replays every logged frame with
`seq > N`, then live-tails while the pump is still running. Response headers:

  - `X-Pupa-Replay-Live`:     `1` while a pump is attached to the thread, else `0`
  - `X-Pupa-Replay-Next-Seq`: the next sequence number the log will assign
  - `X-Pupa-Replay-Oldest-Seq`: oldest seq still buffered (ring may have evicted)

An unknown thread re-attaches to nothing: `204 No Content` (client should give
up cleanly and fall back to normal sends).

## Completed-turn snapshot (catch-up after app kill)

The log is retained for `PUPA_SSE_REPLAY_TTL` seconds (default 21600 = 6h) after the
run finishes. Re-attaching with `after_seq = -1` therefore replays the whole
turn — including the final assistant message and `RUN_FINISHED` — which *is*
the phase-1 "completed-turn snapshot": an app relaunched after being killed
mid-run catches up from the buffer without any delta having streamed live.

## Deliberate phase-1 limits

  - In-memory only: a backend restart drops the buffers (accepted for now;
    persisting the log is future work).
  - Client disconnect no longer cancels an in-flight LangGraph run — the run
    finishes into the buffer. That is exactly the background-survival goal;
    runaway cost is bounded by the run itself ending and the ring caps below.

## Env knobs

  - `PUPA_SSE_REPLAY_TTL`         seconds an idle log survives (default 21600 = 6h; <= 0 disables replay entirely)
  - `PUPA_SSE_REPLAY_MAX_EVENTS`  ring cap per thread, frames (default 4096)
  - `PUPA_SSE_REPLAY_MAX_BYTES`   ring cap per thread, bytes  (default 8 MiB)

## Middleware ordering (load-bearing)

Add this middleware BEFORE `SSEKeepAliveMiddleware` in `app.py` (first added =
innermost). Stack: api_key → keep-alive → replay → handler. The pump then sees
only real agent frames (keep-alive comments are injected *outside*, onto the
tail response — so an idle attached/re-attached client still gets heartbeats),
and a re-attach short-circuit still inherits keep-alives for free.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any, AsyncIterator

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger("uvicorn.error")

_SSE_MEDIA_TYPE = "text/event-stream"

HEADER_LIVE = "X-Pupa-Replay-Live"
HEADER_NEXT_SEQ = "X-Pupa-Replay-Next-Seq"
HEADER_OLDEST_SEQ = "X-Pupa-Replay-Oldest-Seq"

# Idle-log retention when PUPA_SSE_REPLAY_TTL is unset. 6h so an app killed
# mid-run can still catch up from the buffer hours later.
_DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 21600


def _ttl() -> float:
    raw = os.getenv("PUPA_SSE_REPLAY_TTL")
    if raw is None:
        return float(_DEFAULT_TTL_SECONDS)
    try:
        return float(raw)
    except ValueError:
        return float(_DEFAULT_TTL_SECONDS)


def _max_events() -> int:
    try:
        return int(os.getenv("PUPA_SSE_REPLAY_MAX_EVENTS", "4096"))
    except ValueError:
        return 4096


def _max_bytes() -> int:
    try:
        return int(os.getenv("PUPA_SSE_REPLAY_MAX_BYTES", str(8 * 1024 * 1024)))
    except ValueError:
        return 8 * 1024 * 1024


def _is_sse(response) -> bool:
    if getattr(response, "media_type", None) == _SSE_MEDIA_TYPE:
        return True
    content_type = response.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip() == _SSE_MEDIA_TYPE


# ---------------------------------------------------------------------------
# Frame splitting
# ---------------------------------------------------------------------------


class FrameSplitter:
    """Accumulate raw bytes and pop complete SSE frames.

    Frames are delimited by a blank line (`\\n\\n` or `\\r\\n\\r\\n`). Whatever
    trails after the last delimiter stays buffered until more bytes arrive;
    `flush()` returns it as a final frame when the stream ends mid-frame.
    """

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf += chunk
        frames: list[bytes] = []
        while True:
            idx_n = self._buf.find(b"\n\n")
            idx_rn = self._buf.find(b"\r\n\r\n")
            if idx_n == -1 and idx_rn == -1:
                break
            if idx_rn != -1 and (idx_n == -1 or idx_rn < idx_n):
                frame, self._buf = self._buf[:idx_rn], self._buf[idx_rn + 4 :]
            else:
                frame, self._buf = self._buf[:idx_n], self._buf[idx_n + 2 :]
            if frame:
                frames.append(frame)
        return frames

    def flush(self) -> bytes | None:
        frame, self._buf = self._buf.strip(b"\r\n"), b""
        return frame or None


def _is_comment(frame: bytes) -> bool:
    """True when every line of the frame is an SSE comment (`:` prefix)."""
    return all(line.startswith(b":") for line in frame.splitlines() if line)


# ---------------------------------------------------------------------------
# Replay log
# ---------------------------------------------------------------------------


class ReplayLog:
    """Per-thread ring buffer of sequenced SSE frames plus live-tail support.

    Deliberately lock-free: every mutation is synchronous on the single event
    loop, and tails wait on their own `asyncio.Event` rather than a shared
    `Condition`. A Condition would have to re-acquire its lock inside the
    cancellation path of `wait()`, which deadlocks when a tail generator is
    `aclose()`d on client disconnect while holding/awaiting the lock.
    """

    def __init__(self, thread_id: str, max_events: int, max_bytes: int) -> None:
        self.thread_id = thread_id
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.frames: deque[tuple[int, bytes]] = deque()
        self.bytes_total = 0
        self.next_seq = 0
        self.live_pumps = 0
        self._waiters: set[asyncio.Event] = set()
        self.last_activity = time.monotonic()

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    @property
    def oldest_seq(self) -> int:
        return self.frames[0][0] if self.frames else self.next_seq

    def _wake(self) -> None:
        for ev in self._waiters:
            ev.set()

    def append(self, frame: bytes) -> int:
        """Stamp `frame` with the next seq, store it, wake tails. Returns the seq."""
        seq = self.next_seq
        self.next_seq += 1
        stamped = b"id: %d\n%s\n\n" % (seq, frame)
        self.frames.append((seq, stamped))
        self.bytes_total += len(stamped)
        while self.frames and (
            len(self.frames) > self.max_events or self.bytes_total > self.max_bytes
        ):
            _, evicted = self.frames.popleft()
            self.bytes_total -= len(evicted)
        self.touch()
        self._wake()
        return seq

    def mark_pump_started(self) -> None:
        self.live_pumps += 1
        self.touch()
        self._wake()

    def mark_pump_done(self) -> None:
        self.live_pumps = max(0, self.live_pumps - 1)
        self.touch()
        self._wake()

    async def tail(self, after_seq: int) -> AsyncIterator[bytes]:
        """Replay frames with seq > `after_seq`, then live-tail while a pump runs.

        Ends when no pump is live and everything buffered has been yielded.
        Closing the generator (client disconnect) affects nothing but this tail.
        """
        cursor = after_seq
        while True:
            batch = [(s, f) for s, f in self.frames if s > cursor]
            if batch:
                for seq, frame in batch:
                    cursor = seq
                    yield frame
                self.touch()
                continue
            if self.live_pumps == 0:
                return
            ev = asyncio.Event()
            self._waiters.add(ev)
            try:
                await ev.wait()
            finally:
                self._waiters.discard(ev)


# Module-level registry keyed by thread_id, plus lazy TTL sweeping.
_LOGS: dict[str, ReplayLog] = {}


def _sweep(ttl: float) -> None:
    now = time.monotonic()
    stale = [
        tid
        for tid, log in _LOGS.items()
        if log.live_pumps == 0 and now - log.last_activity > ttl
    ]
    for tid in stale:
        del _LOGS[tid]
        logger.info("sse_replay: evicted idle replay log thread_id=%s", tid)


def _get_or_create(thread_id: str) -> ReplayLog:
    log = _LOGS.get(thread_id)
    if log is None:
        log = ReplayLog(thread_id, max_events=_max_events(), max_bytes=_max_bytes())
        _LOGS[thread_id] = log
    return log


def reset_logs() -> None:
    """Test hook: drop all replay state."""
    _LOGS.clear()


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def _extract(body: bytes) -> tuple[str | None, int | None]:
    """Pull (thread_id, reattach_after_seq) out of a `POST /` JSON body.

    `after_seq` is None when the request is a normal run (no
    `forwardedProps.command.reattach`).
    """
    try:
        payload: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    thread_id = payload.get("threadId") or payload.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        return None, None
    fp = payload.get("forwardedProps") or payload.get("forwarded_props")
    command = fp.get("command") if isinstance(fp, dict) else None
    reattach = command.get("reattach") if isinstance(command, dict) else None
    if not isinstance(reattach, dict):
        return thread_id, None
    raw = reattach.get("after_seq", reattach.get("afterSeq", -1))
    try:
        after_seq = int(raw)
    except (TypeError, ValueError):
        after_seq = -1
    return thread_id, after_seq


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


async def _pump(upstream, log: ReplayLog) -> None:
    """Drain the handler's body iterator into the log, come what may.

    Runs detached from the HTTP response so a client disconnect cannot kill
    the agent run. Comment-only frames (keep-alives, should the ordering ever
    put one inside) are forwarded to nobody — they carry no replay value.
    """
    splitter = FrameSplitter()
    try:
        async for chunk in upstream:
            if isinstance(chunk, str):  # StreamingResponse allows str chunks
                chunk = chunk.encode("utf-8")
            for frame in splitter.feed(chunk):
                if _is_comment(frame):
                    continue
                log.append(frame)
        trailing = splitter.flush()
        if trailing is not None and not _is_comment(trailing):
            log.append(trailing)
    except asyncio.CancelledError:  # app shutdown
        raise
    except Exception:  # noqa: BLE001 — upstream (agent) error; tails end cleanly
        logger.exception("sse_replay: pump failed thread_id=%s", log.thread_id)
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.debug("sse_replay: upstream aclose failed", exc_info=True)
        log.mark_pump_done()


def _replay_headers(log: ReplayLog) -> dict[str, str]:
    return {
        HEADER_LIVE: "1" if log.live_pumps > 0 else "0",
        HEADER_NEXT_SEQ: str(log.next_seq),
        HEADER_OLDEST_SEQ: str(log.oldest_seq),
    }


def _is_run_path(path: str) -> bool:
    """True for a harness run endpoint: `/` (default alias) or `/harnesses/{id}`.

    The replay log is keyed by thread_id, and a thread only ever talks to one
    harness, so sharing the log across paths is safe.
    """
    return path == "/" or path.startswith("/harnesses/")


class SSEReplayMiddleware(BaseHTTPMiddleware):
    """Detach harness-run SSE bodies into per-thread replay logs; serve re-attaches.

    Covers `POST /` and `POST /harnesses/{id}` (a thread pins to one harness, so
    the thread-keyed log is unambiguous)."""

    async def dispatch(self, request: Request, call_next):
        ttl = _ttl()
        if ttl <= 0 or request.method != "POST" or not _is_run_path(request.url.path):
            return await call_next(request)

        body = await request.body()  # cached by Starlette; downstream re-reads fine
        thread_id, after_seq = _extract(body)
        _sweep(ttl)

        # --- Re-attach: short-circuit, never reaches an agent loop. ---------
        if thread_id is not None and after_seq is not None:
            log = _LOGS.get(thread_id)
            if log is None:
                return Response(status_code=204)
            logger.info(
                "sse_replay: reattach thread_id=%s after_seq=%d (next=%d live=%d)",
                thread_id, after_seq, log.next_seq, log.live_pumps,
            )
            return StreamingResponse(
                log.tail(after_seq),
                media_type=_SSE_MEDIA_TYPE,
                headers=_replay_headers(log),
            )

        # --- Normal run: detach the handler stream into the log. ------------
        response = await call_next(request)
        if thread_id is None or not _is_sse(response):
            return response

        log = _get_or_create(thread_id)
        start_seq = log.next_seq - 1  # tail everything this run appends
        log.mark_pump_started()
        asyncio.ensure_future(_pump(response.body_iterator, log))

        return StreamingResponse(
            log.tail(start_seq),
            status_code=response.status_code,
            headers={**dict(response.headers), **_replay_headers(log)},
            media_type=response.media_type or _SSE_MEDIA_TYPE,
            background=response.background,
        )
