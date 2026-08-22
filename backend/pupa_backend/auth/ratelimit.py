"""Per-client throttling for the pairing endpoints.

`/auth/pair` is the only unauthenticated write route on the backend: the
bootstrap code *is* the credential, so it can't sit behind a token. Codes are
single-use with a short TTL over a 31^8 space, which makes guessing
impractical — but nothing stopped a caller from trying at full speed.

**Only failures are charged.** That's the whole design, and it follows from
what these two routes are. Mechanically the charge goes on at entry and is
refunded when the response says the caller was legitimate — checking first and
charging after `call_next` would let any number of concurrent requests clear
the check before the first is recorded, capping guesses at the connection
count instead of at the limit.

- On `/auth/pair` a legitimate device sends exactly one request, it succeeds,
  and the success consumes the code so it can't even be replayed. Every
  *failed* request is therefore a guess. Charging successes would spend the
  budget only on the people entitled to it.
- `/auth/pair/begin` is operator-only. Throttling a caller who already
  presented `PUPA_API_KEY` protects nothing — that key grants everything
  anyway — while blocking an operator pairing a batch of devices. What is
  worth throttling there is *wrong-key* attempts.

The one thing the provisional charge does cost: more than `limit` requests
genuinely *in flight at once* on the same bucket will 429 even when they would
all have succeeded, because the budget is held for the duration of each. A
`make pair` loop is sequential and never sees it; firing 15 concurrent mints at
one backend does. Retrying works immediately — the charges come back off as
those requests finish, so the `Retry-After` (computed from the in-flight
charges) is an upper bound, not a wait the caller has to serve.

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
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .devices import truthy
from .proxy import forwarded_values, trust_forwarded_headers

# FAILED attempts per client per minute — successful requests are free.
# `/auth/pair` is the public one and a human types one code once, so a
# legitimate caller never sees this; five wrong codes in a minute is a guesser.
PAIR_EXCHANGE_LIMIT = 5
# Wrong `PUPA_API_KEY` (401) or a device token trying to mint (403).
PAIR_BEGIN_LIMIT = 10
WINDOW_SECONDS = 60.0
# A bucket key is attacker-influenced (it can be a forwarded address), and
# `under_limit` is asked on every request including the ones never charged. So
# the map is bounded on both axes: keys are truncated, empty buckets are
# dropped as soon as they age out, and the oldest buckets are evicted if it
# somehow still grows past a ceiling no legitimate deployment approaches.
MAX_KEY_CHARS = 64
MAX_TRACKED_KEYS = 10_000
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
        self._hits: dict[str, deque[float]] = {}
        self._now = now

    def _live_hits(self, key: str, window: float) -> deque[float]:
        """Hits still inside the window. Read-only: never creates a bucket, so
        asking about an unknown key costs nothing and leaks nothing."""
        hits = self._hits.get(key)
        if hits is None:
            return deque()
        cutoff = self._now() - window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            # Drop it now rather than retaining an empty deque per address
            # ever seen — that map is keyed on attacker-supplied values.
            del self._hits[key]
        return hits

    def under_limit(self, key: str, limit: int, window: float = WINDOW_SECONDS) -> bool:
        """Whether this key still has budget. A question, not a debit — the
        middleware charges separately, via `record`."""
        return len(self._live_hits(key, window)) < limit

    def record(self, key: str, window: float = WINDOW_SECONDS) -> float:
        """Charge one attempt against this key. Returns the timestamp written,
        which `refund` takes back if the response says the caller was
        legitimate."""
        live = self._live_hits(key, window)
        if len(self._hits) >= MAX_TRACKED_KEYS and key not in self._hits:
            # Only reachable under a distributed flood. Clearing the map would
            # hand every caller their outstanding budget back at once, so a
            # flood spread over enough addresses could keep the limiter
            # switched off for everyone simply by continuing to run. Evict the
            # buckets closest to ageing out instead — least *recently* charged,
            # which is not insertion order: `record` re-assigns an existing key
            # in place, so a guesser that started before the flood sits at the
            # front of the map and is exactly the bucket that must survive it.
            by_last_hit = sorted(self._hits, key=lambda k: self._hits[k][-1])
            for stale in by_last_hit[:MAX_TRACKED_KEYS // 10]:
                del self._hits[stale]
        stamp = self._now()
        live.append(stamp)
        self._hits[key] = live
        return stamp

    def refund(self, key: str, stamp: float, window: float = WINDOW_SECONDS) -> None:
        """Give back the exact hit `record` returned. By value, not "the newest
        one": concurrent requests share a bucket, so popping the tail would
        return a charge that still has a request behind it, and a request that
        outlived the window would take back a hit it never wrote."""
        live = self._live_hits(key, window)
        try:
            live.remove(stamp)
        except ValueError:
            return  # already aged out of the window — nothing to give back
        if not live:
            del self._hits[key]

    def retry_after(self, key: str, window: float = WINDOW_SECONDS) -> int:
        """Whole seconds until the oldest hit in this key's window ages out."""
        hits = self._hits.get(key)
        if not hits:
            return 1
        return max(1, int(hits[0] + window - self._now()) + 1)

    def tracked_keys(self) -> int:
        """For tests — how many buckets are being retained."""
        return len(self._hits)

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

    The header is only read when `trust_forwarded_headers()` says something in
    front actually writes it (see `proxy.py`). Otherwise it is a client-supplied
    string and believing it would let one host rotate the header per request to
    get an unlimited number of buckets — no limit at all.

    When trusted, the **rightmost** entry wins: a caller can forge the header,
    but the proxy *appends* the address it actually saw, so that entry is the
    only one it wrote. This assumes a single appending hop; behind a chain of
    two or more, the rightmost is the previous proxy and callers collapse into
    one bucket. Safe direction — failures are all that's ever charged — but it
    means a multi-hop deployment wants `PUPA_TRUSTED_PROXY` weighed
    deliberately.
    """
    if trust_forwarded_headers():
        hops = forwarded_values(request.headers, "x-forwarded-for")
        if hops:
            return hops[-1][:MAX_KEY_CHARS]
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    return (host or "unknown")[:MAX_KEY_CHARS]


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

    Budget is spent by *failed* attempts only (see the module docstring). The
    charge goes on before the request and comes back off once the status code
    says the caller had a valid credential — charging only afterwards would
    let concurrent requests all clear the check against the same empty bucket.

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

    charge = limiter.record(key)
    try:
        response = await call_next(request)
    except BaseException:
        # A crash downstream is a server bug, not a guess — don't let it spend
        # the caller's budget.
        limiter.refund(key, charge)
        raise
    # `require_https_middleware` sits inside this one and also answers 403.
    # That's a transport misconfiguration, not a guess at a credential — don't
    # spend a legitimate device's budget on it.
    if response.status_code not in FAILURE_STATUSES or getattr(
        request.state, "transport_refused", False
    ):
        limiter.refund(key, charge)
    return response
