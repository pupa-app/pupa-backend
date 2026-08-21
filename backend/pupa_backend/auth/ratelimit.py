"""Per-client throttling for the pairing endpoints.

`/auth/pair` is the only unauthenticated write route on the backend: the
bootstrap code *is* the credential, so it can't sit behind a token. Codes are
single-use with a short TTL over a 31^8 space, which makes guessing
impractical — but nothing stopped a caller from trying at full speed, or from
flooding the route outright. This adds the missing per-client ceiling.

Hand-rolled rather than `slowapi`: the surface is two routes, the process is
single-worker (`app.py` runs uvicorn with no `workers=`), and the useful part
of a limiter here is the key function, which we'd have to write either way —
`get_remote_address` reads `request.client.host` and walks straight into the
loopback trap documented on `client_key`. If workers are ever added, this
needs a shared backing store.
"""

import os
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .devices import truthy

# Attempts per client per minute. `/auth/pair` is the public one, so it's the
# tighter of the two; a human pairing a device types one code, once.
PAIR_EXCHANGE_LIMIT = 5
PAIR_BEGIN_LIMIT = 10
# Backstop across all clients, so a botnet can't sidestep the per-client cap
# by spreading the attempts out.
PAIR_GLOBAL_LIMIT = 60
WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    """Fixed-capacity-per-window counter, keyed by an arbitrary string.

    Keeps hit timestamps per key and drops the ones that have aged out on
    every check, so the window really slides — a fixed bucket would let a
    caller spend the whole allowance at the end of one window and again at the
    start of the next.
    """

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._now = now

    def allow(self, key: str, limit: int, window: float = WINDOW_SECONDS) -> bool:
        """Record a hit and report whether it's within the allowance."""
        now = self._now()
        hits = self._hits[key]
        cutoff = now - window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True

    def retry_after(self, key: str, window: float = WINDOW_SECONDS) -> int:
        """Whole seconds until the oldest hit in this key's window ages out."""
        hits = self._hits.get(key)
        if not hits:
            return 1
        return max(1, int(hits[0] + window - self._now()) + 1)

    def clear(self) -> None:
        self._hits.clear()


_limiter = SlidingWindowLimiter()


def get_limiter() -> SlidingWindowLimiter:
    return _limiter


def reset_for_testing() -> SlidingWindowLimiter:
    _limiter.clear()
    return _limiter


def client_key(request: Request) -> str:
    """Identify the caller for bucketing.

    **The loopback trap:** every Pupa transport terminates TLS in front of a
    loopback-bound listener — Tailscale serve, Cloudflare tunnel, Railway. So
    `request.client.host` is `127.0.0.1` for *all* remote callers, and
    bucketing on it would throttle every user as one. `X-Forwarded-For` is
    what distinguishes them.

    A caller can forge that header, but the trusted proxy **appends** the
    address it actually saw, so the **rightmost** entry is the only one the
    proxy wrote — read that, not the leftmost. With no proxy in front (bound
    to a real interface), there's no header and the peer address is honest.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1]
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    return host or "unknown"


def _limit_for(path: str) -> int | None:
    if path == "/auth/pair":
        return PAIR_EXCHANGE_LIMIT
    if path == "/auth/pair/begin":
        return PAIR_BEGIN_LIMIT
    return None


async def rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Throttle the pairing routes. Mounted **outermost** so it runs before any
    auth work and regardless of the outcome — pairing is pre-auth.

    `POST /` is deliberately not throttled: a dropped SSE socket re-attaches
    there (see `SSEReplayMiddleware`), so a per-IP cap would break exactly the
    flaky-network case that machinery exists for. It's authenticated, and now
    scope-gated, which is the ceiling that fits it.
    """
    if request.method != "POST":
        return await call_next(request)
    limit = _limit_for(request.url.path)
    if limit is None:
        return await call_next(request)
    if truthy(os.getenv("PUPA_RATE_LIMIT_DISABLED")):
        return await call_next(request)

    limiter = get_limiter()
    key = f"{request.url.path}:{client_key(request)}"
    global_key = f"{request.url.path}:*"
    if not limiter.allow(key, limit) or not limiter.allow(global_key, PAIR_GLOBAL_LIMIT):
        blocked = key if limiter.retry_after(key) > 1 else global_key
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many pairing attempts. Try again shortly."},
            headers={"Retry-After": str(limiter.retry_after(blocked))},
        )
    return await call_next(request)
