"""Per-client throttling for the pairing endpoints.

`/auth/pair` is the only unauthenticated write route on the backend: the
bootstrap code *is* the credential, so it can't sit behind a token. Codes are
single-use with a short TTL over a 31^8 space, which makes guessing
impractical — but nothing stopped a caller from trying at full speed.

**Only failures are charged.** That's the whole design, and it follows from
what these two routes are:

- On `/auth/pair` a legitimate device sends exactly one request, it succeeds,
  and the success consumes the code so it can't even be replayed. Every
  *failed* request is therefore a guess. Charging successes would spend the
  budget only on the people entitled to it.
- `/auth/pair/begin` is operator-only. Throttling a caller who already
  presented `PUPA_API_KEY` protects nothing — that key grants everything
  anyway — while blocking an operator pairing a batch of devices. What is
  worth throttling there is *wrong-key* attempts.

**Per-client only, no global backstop.** A shared bucket that blocks is a
denial of service with extra steps: a stranger drains it with junk and every
legitimate caller gets 429 regardless of the credential they hold. Charging
only failures slows the draining but doesn't change the outcome. The shared
cap's one purpose was a botnet spreading guesses over many addresses, and the
arithmetic already covers that — 31^8 is 8.5e11 codes, each single-use and
alive for five minutes, so even 10k addresses at five guesses a minute get
nowhere. A limit that can be turned against the people it protects buys less
than it costs.

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

# FAILED attempts per client per minute — successful requests are free.
# `/auth/pair` is the public one and a human types one code once, so a
# legitimate caller never sees this; five wrong codes in a minute is a guesser.
PAIR_EXCHANGE_LIMIT = 5
# Wrong `PUPA_API_KEY` (401) or a device token trying to mint (403).
PAIR_BEGIN_LIMIT = 10
WINDOW_SECONDS = 60.0
# Response codes that mean "this caller did not present a valid credential".
# 422 is deliberately absent: a malformed body is a client bug, not a guess at
# a secret, and charging it would let a buggy client lock itself out.
FAILURE_STATUSES = frozenset({401, 403, 404})


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

    def _prune(self, key: str, window: float) -> deque[float]:
        hits = self._hits[key]
        cutoff = self._now() - window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        return hits

    def under_limit(self, key: str, limit: int, window: float = WINDOW_SECONDS) -> bool:
        """Whether this key still has budget. A question, not a debit — the
        middleware asks it on every request but only charges the failures."""
        return len(self._prune(key, window)) < limit

    def record(self, key: str, window: float = WINDOW_SECONDS) -> None:
        """Charge one failure against this key."""
        self._prune(key, window).append(self._now())

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


def _too_many(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many failed pairing attempts. Try again shortly."},
        headers={"Retry-After": str(retry_after)},
    )


async def rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Throttle the pairing routes. Mounted **outermost** so it runs before any
    auth work and regardless of the outcome — pairing is pre-auth.

    Budget is spent by *failed* attempts only (see the module docstring), so
    the check runs before the request and the charge after it, once the status
    code says whether the caller had a valid credential.

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

    if not limiter.under_limit(key, limit):
        return _too_many(limiter.retry_after(key))

    response = await call_next(request)
    if response.status_code in FAILURE_STATUSES:
        limiter.record(key)
    return response
