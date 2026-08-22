"""Abuse throttling on the pairing endpoints.

`/auth/pair` is the one unauthenticated write route on the backend, so it's
the only surface a stranger can hammer. These cover the limiter itself, the
proxy-aware client key it buckets on, and that it's wired outermost.
"""


from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, rate_limit_middleware, router as auth_router
from pupa_backend.auth.devices import reset_for_testing as reset_devices
from pupa_backend.auth.pairing import reset_for_testing as reset_pairing
from pupa_backend.auth.ratelimit import (
    PAIR_BEGIN_LIMIT,
    PAIR_EXCHANGE_LIMIT,
    SlidingWindowLimiter,
    client_key,
    get_limiter,
    reset_for_testing as reset_limiter,
)


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.delenv("PUPA_RATE_LIMIT_DISABLED", raising=False)
    # These cases exercise the proxied deployments, where something in front
    # writes X-Forwarded-For. The direct-bind case is covered separately below.
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")
    reset_devices(tmp_path / "devices.json")
    reset_pairing()
    reset_limiter()
    app = FastAPI()
    # Same order as `app.py`: the limiter is added last, so it sits outermost
    # and throttles before any auth work happens.
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(rate_limit_middleware)
    app.include_router(auth_router, prefix="/auth")
    return app


# ---------------------------------------------------------------------------
# SlidingWindowLimiter
# ---------------------------------------------------------------------------


def test_limiter_blocks_once_the_budget_is_spent() -> None:
    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    for _ in range(3):
        assert limiter.under_limit("ip", limit=3)
        limiter.record("ip")
    assert not limiter.under_limit("ip", limit=3)


def test_peeking_does_not_spend_budget() -> None:
    """`under_limit` is a question, not a debit — the middleware asks it on
    every request but only charges the ones that fail."""
    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    for _ in range(50):
        assert limiter.under_limit("ip", limit=1)
    limiter.record("ip")
    assert not limiter.under_limit("ip", limit=1)


def test_limiter_forgets_hits_older_than_the_window() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(now=lambda: now[0])
    for _ in range(3):
        limiter.record("ip")
    assert not limiter.under_limit("ip", limit=3)
    now[0] = 61.0  # the three hits have aged out of the 60 s window
    assert limiter.under_limit("ip", limit=3)


def test_limiter_buckets_are_independent_per_key() -> None:
    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    limiter.record("a")
    assert not limiter.under_limit("a", limit=1)
    assert limiter.under_limit("b", limit=1)


# ---------------------------------------------------------------------------
# client_key — the loopback trap
# ---------------------------------------------------------------------------


class _Req:
    def __init__(self, peer: str | None, xff: str | None = None) -> None:
        self.client = type("C", (), {"host": peer})() if peer else None
        self.headers = {"x-forwarded-for": xff} if xff else {}


@pytest.fixture
def trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")


@pytest.fixture
def untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_TRUSTED_PROXY", raising=False)
    monkeypatch.delenv("PUPA_CONNECTIVITY", raising=False)


def test_client_key_prefers_the_forwarded_client(trusted: None) -> None:
    """Every transport mode (Tailscale serve, Cloudflare tunnel, Railway)
    terminates in front of a loopback-bound listener, so `client.host` is
    127.0.0.1 for *every* remote caller. Bucketing on it would put the whole
    internet in one bucket."""
    assert client_key(_Req("127.0.0.1", "203.0.113.7")) == "203.0.113.7"


def test_client_key_takes_the_rightmost_forwarded_entry(trusted: None) -> None:
    """A caller can forge `X-Forwarded-For`; the trusted proxy *appends* what
    it actually saw. The rightmost entry is the only one it wrote."""
    assert client_key(_Req("127.0.0.1", "1.1.1.1, 2.2.2.2, 203.0.113.7")) == "203.0.113.7"


def test_client_key_falls_back_to_the_peer_without_a_proxy(trusted: None) -> None:
    assert client_key(_Req("198.51.100.4")) == "198.51.100.4"


def test_client_key_survives_a_missing_peer(trusted: None) -> None:
    assert client_key(_Req(None)) == "unknown"


# ---------------------------------------------------------------------------
# Throttling charges failures, not legitimate use
# ---------------------------------------------------------------------------


def _bad_code(client: TestClient, xff: str = "203.0.113.7"):
    return client.post(
        "/auth/pair",
        json={"code": "NOSUCHCO", "label": "x"},
        headers={"X-Forwarded-For": xff},
    )


def test_wrong_codes_are_throttled(app: FastAPI) -> None:
    """The 8-char code IS the credential on this route, so every failure is a
    guess."""
    codes = [_bad_code(TestClient(app)).status_code for _ in range(PAIR_EXCHANGE_LIMIT + 1)]
    assert codes[:-1] == [404] * PAIR_EXCHANGE_LIMIT
    assert codes[-1] == 429


def test_a_successful_pairing_costs_nothing(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the whole redesign: a legitimate device pairs in one
    request and must never be pushed toward the limit for doing so. Spend the
    budget down to its last unit, then prove a real pairing still goes
    through — and still doesn't tip it over."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    xff = {"X-Forwarded-For": "203.0.113.7"}

    for _ in range(PAIR_EXCHANGE_LIMIT - 1):
        assert _bad_code(client).status_code == 404

    begin = client.post(
        "/auth/pair/begin", json={}, headers={"Authorization": "Bearer k", **xff}
    ).json()
    ok = client.post(
        "/auth/pair", json={"code": begin["code"], "label": "phone"}, headers=xff
    )
    assert ok.status_code == 200

    # The success didn't spend anything: one failure of headroom remains.
    assert _bad_code(client).status_code == 404
    assert _bad_code(client).status_code == 429


def test_authenticated_minting_is_not_throttled(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/auth/pair/begin` is operator-only. Throttling a caller who already
    presented PUPA_API_KEY protects nothing — they hold the credential that
    grants everything — and blocks an operator pairing a batch of devices."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    headers = {"Authorization": "Bearer k", "X-Forwarded-For": "203.0.113.7"}
    for _ in range(PAIR_BEGIN_LIMIT * 3):
        assert client.post("/auth/pair/begin", json={}, headers=headers).status_code == 200


def test_wrong_operator_keys_are_throttled(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What throttling this route is actually for: guessing PUPA_API_KEY."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    headers = {"Authorization": "Bearer wrong", "X-Forwarded-For": "203.0.113.7"}
    codes = [
        client.post("/auth/pair/begin", json={}, headers=headers).status_code
        for _ in range(PAIR_BEGIN_LIMIT + 1)
    ]
    assert codes[:-1] == [401] * PAIR_BEGIN_LIMIT
    assert codes[-1] == 429


def test_a_rejected_device_token_is_throttled(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device token gets 403 here (devices can't mint devices). That's a
    failure, so it's charged — a compromised device can't grind against the
    route hunting for a way in."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin", json={}, headers={"Authorization": "Bearer k"}
    ).json()
    token = client.post(
        "/auth/pair", json={"code": begin["code"], "label": "phone"}
    ).json()["token"]

    headers = {"Authorization": f"Bearer {token}", "X-Forwarded-For": "198.51.100.5"}
    codes = [
        client.post("/auth/pair/begin", json={}, headers=headers).status_code
        for _ in range(PAIR_BEGIN_LIMIT + 1)
    ]
    assert codes[:-1] == [403] * PAIR_BEGIN_LIMIT
    assert codes[-1] == 429


def test_a_forged_header_cannot_buy_new_buckets(untrusted: None) -> None:
    """The bypass this trust check exists to close. On a direct bind — the
    LAN/offline default, `0.0.0.0`, nothing in front — `X-Forwarded-For` is
    just a string the caller chose. Believing it would let one host rotate it
    per request for an unlimited supply of buckets, i.e. no limit at all."""
    a = client_key(_Req("198.51.100.4", "203.0.113.1"))
    b = client_key(_Req("198.51.100.4", "203.0.113.2"))
    assert a == b == "198.51.100.4"


def test_forged_headers_do_not_escape_the_limit(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: same socket, a fresh forged address every request, and the
    limit still bites once the trust flag is off."""
    monkeypatch.delenv("PUPA_TRUSTED_PROXY", raising=False)
    monkeypatch.delenv("PUPA_CONNECTIVITY", raising=False)
    client = TestClient(app)
    codes = [
        _bad_code(client, xff=f"203.0.113.{i}").status_code
        for i in range(PAIR_EXCHANGE_LIMIT + 3)
    ]
    assert 429 in codes, "rotating a forged X-Forwarded-For bypassed the limit"


def test_key_length_is_capped(trusted: None) -> None:
    """The key is attacker-supplied when forwarded headers are trusted."""
    assert len(client_key(_Req("127.0.0.1", "x" * 5000))) <= 64


def test_throttling_is_per_client(app: FastAPI) -> None:
    client = TestClient(app)
    for _ in range(PAIR_EXCHANGE_LIMIT):
        assert _bad_code(client).status_code == 404
    assert _bad_code(client, xff="198.51.100.9").status_code == 404


def test_no_amount_of_third_party_abuse_blocks_a_legitimate_pairing(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property that ruled out a shared/global bucket. Whatever strangers
    do, from however many addresses, a caller holding a real credential is
    unaffected — buckets are per-client, so there is nothing shared to drain.
    """
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)

    for i in range(500):
        _bad_code(client, xff=f"203.0.113.{i % 250}")

    begin = client.post(
        "/auth/pair/begin", json={}, headers={"Authorization": "Bearer k"}
    ).json()
    ok = client.post(
        "/auth/pair",
        json={"code": begin["code"], "label": "phone"},
        headers={"X-Forwarded-For": "198.51.100.77"},
    )
    assert ok.status_code == 200, "third-party abuse blocked a real pairing"


def test_rate_limit_can_be_disabled_for_dev(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUPA_RATE_LIMIT_DISABLED", "1")
    client = TestClient(app)
    codes = [_bad_code(client).status_code for _ in range(PAIR_EXCHANGE_LIMIT + 3)]
    assert 429 not in codes


def test_throttled_response_says_how_long_to_wait(app: FastAPI) -> None:
    client = TestClient(app)
    for _ in range(PAIR_EXCHANGE_LIMIT):
        _bad_code(client)
    resp = _bad_code(client)
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") is not None


def test_unrelated_routes_are_not_throttled(app: FastAPI) -> None:
    client = TestClient(app)
    for _ in range(PAIR_EXCHANGE_LIMIT + 5):
        assert client.get("/auth/config").status_code == 200


# ---------------------------------------------------------------------------
# Bucket map stays bounded
# ---------------------------------------------------------------------------


def test_asking_about_an_unknown_key_does_not_create_a_bucket() -> None:
    """`under_limit` runs on every request, including ones never charged. If
    the question allocated, a flood of successes would grow the map at request
    rate."""
    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    for i in range(1000):
        limiter.under_limit(f"client-{i}", limit=5)
    assert limiter.tracked_keys() == 0


def test_expired_buckets_are_dropped_not_retained_empty() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(now=lambda: now[0])
    for i in range(500):
        limiter.record(f"client-{i}")
    assert limiter.tracked_keys() == 500
    now[0] = 61.0
    for i in range(500):
        limiter.under_limit(f"client-{i}", limit=5)
    assert limiter.tracked_keys() == 0, "empty buckets retained after the window"


def test_the_map_cannot_grow_without_bound() -> None:
    from pupa_backend.auth.ratelimit import MAX_TRACKED_KEYS

    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    for i in range(MAX_TRACKED_KEYS + 500):
        limiter.record(f"client-{i}")
    assert limiter.tracked_keys() <= MAX_TRACKED_KEYS


# ---------------------------------------------------------------------------
# Concurrency: the check and the charge must not straddle an await
# ---------------------------------------------------------------------------


async def test_concurrent_guesses_cannot_exceed_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checking the budget before `call_next` and charging after it caps
    guesses at the *connection count*, not at the limit: every request in
    flight evaluates `under_limit` against a bucket nothing has been written
    to yet. `/auth/pair` really does await in the middle (the pairing lock,
    then the device store's file I/O), so this is reachable with nothing more
    exotic than parallel requests.
    """
    import asyncio

    from httpx import ASGITransport, AsyncClient

    monkeypatch.delenv("PUPA_RATE_LIMIT_DISABLED", raising=False)
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")
    reset_limiter()

    gate = asyncio.Event()
    app = FastAPI()

    @app.post("/auth/pair")
    async def pair() -> JSONResponse:
        # Every admitted request parks here, so they are all in flight at once
        # — the window the old ordering left open.
        await gate.wait()
        return JSONResponse(status_code=404, content={"detail": "no such code"})

    app.middleware("http")(rate_limit_middleware)

    headers = {"X-Forwarded-For": "203.0.113.9"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        attempts = [
            asyncio.create_task(client.post("/auth/pair", json={}, headers=headers))
            for _ in range(PAIR_EXCHANGE_LIMIT * 4)
        ]
        # Let every request reach the middleware (and the gate) before any of
        # them is allowed to finish.
        for _ in range(50):
            await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(*attempts)

    admitted = [r for r in results if r.status_code != 429]
    assert len(admitted) <= PAIR_EXCHANGE_LIMIT, (
        f"{len(admitted)} concurrent guesses were admitted against a budget of "
        f"{PAIR_EXCHANGE_LIMIT}"
    )


def test_a_refund_gives_back_the_callers_own_charge() -> None:
    """The middleware charges on entry and refunds when the response says the
    caller was legitimate. A refund must return that caller's own charge, not
    forgive an older guess."""
    now = [0.0]
    limiter = SlidingWindowLimiter(now=lambda: now[0])
    limiter.record("ip")                    # an earlier guess
    now[0] = 10.0
    charge = limiter.record("ip")           # the request being refunded
    limiter.refund("ip", charge)
    assert limiter.under_limit("ip", limit=2)
    assert not limiter.under_limit("ip", limit=1), "the older guess was forgiven"


def test_a_refund_cannot_take_back_a_concurrent_requests_charge() -> None:
    """Requests share a bucket, so refunding "the newest hit" would hand back a
    charge that still has a request behind it. Refund by value."""
    now = [0.0]
    limiter = SlidingWindowLimiter(now=lambda: now[0])
    mine = limiter.record("ip")             # request A arrives at t=0
    now[0] = 1.0
    limiter.record("ip")                    # request B arrives at t=1, in flight
    limiter.refund("ip", mine)              # A finishes first, legitimately
    assert not limiter.under_limit("ip", limit=1), "B's charge was refunded too"
    # The hit left behind must be *B's*. Popping the tail would leave A's, and
    # the count alone can't tell them apart — the expiry can.
    assert limiter.retry_after("ip") == 61, "the wrong charge was given back"


def test_a_refund_of_an_aged_out_charge_is_harmless() -> None:
    """A request that outlived the window has nothing left to give back — and
    must not take a hit it never wrote."""
    now = [0.0]
    limiter = SlidingWindowLimiter(now=lambda: now[0])
    stale = limiter.record("ip")
    now[0] = 61.0
    fresh = limiter.record("ip")            # a different caller, same bucket
    limiter.refund("ip", stale)
    assert not limiter.under_limit("ip", limit=1), "someone else's charge was taken"
    limiter.refund("ip", fresh)
    assert limiter.tracked_keys() == 0


def test_a_refund_on_an_untouched_key_is_harmless() -> None:
    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    limiter.refund("never-seen", 0.0)
    assert limiter.tracked_keys() == 0


def test_the_ceiling_keeps_the_buckets_that_are_still_being_spent() -> None:
    """Clearing the map at the ceiling would hand every caller their
    outstanding budget back at once — a flood spread over enough addresses
    could keep the limiter switched off for everybody simply by running. Nor
    can eviction go by insertion order: a guesser that started *before* the
    flood is at the front of the map and is exactly the bucket that has to
    survive it.
    """
    from pupa_backend.auth.ratelimit import MAX_TRACKED_KEYS

    now = [0.0]
    limiter = SlidingWindowLimiter(now=lambda: now[0])
    # The guesser is the *first* bucket in the map — insertion order puts it at
    # the front, so evicting by that order would drop the one caller here that
    # is actually being throttled.
    limiter.record("guesser")
    for i in range(MAX_TRACKED_KEYS - 1):
        now[0] += 0.001
        limiter.record(f"flood-{i}")
    # It is still spending, right up to the moment the ceiling is tripped.
    for _ in range(4):
        now[0] += 0.001
        limiter.record("guesser")
    now[0] += 0.001
    limiter.record("one-more-address")

    assert limiter.tracked_keys() <= MAX_TRACKED_KEYS, "the ceiling did not hold"
    assert not limiter.under_limit("guesser", limit=5), (
        "hitting the ceiling forgave a live budget"
    )


# ---------------------------------------------------------------------------
# What the middleware refunds, and what it never charges
# ---------------------------------------------------------------------------


def test_a_transport_refusal_does_not_spend_a_pairing_budget(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plaintext hop is a misconfiguration, not a guess at a credential — an
    operator who forgot the scheme must not burn a real device's budget
    discovering it. `require_https_middleware` is mounted outside the limiter
    so the 403 is returned before the charge is ever made; this mirrors that
    stack (added last = outermost)."""
    from pupa_backend.auth import require_https_middleware

    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    reset_limiter()
    inner = FastAPI()
    inner.middleware("http")(rate_limit_middleware)
    inner.middleware("http")(require_https_middleware)
    inner.include_router(auth_router, prefix="/auth")

    client = TestClient(inner)
    headers = {"X-Forwarded-For": "203.0.113.11"}
    for _ in range(PAIR_EXCHANGE_LIMIT * 3):
        resp = client.post("/auth/pair", json={"code": "NOSUCHCO", "label": "x"}, headers=headers)
        assert resp.status_code == 403, resp.status_code
    assert get_limiter().tracked_keys() == 0, "a transport refusal was charged"


def test_a_crash_downstream_does_not_spend_the_callers_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 500 is a server bug, not a guess. Leaving the entry charge on would
    let a broken deploy lock out the device it is broken for."""
    monkeypatch.delenv("PUPA_RATE_LIMIT_DISABLED", raising=False)
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")
    reset_limiter()

    boom = FastAPI()

    @boom.post("/auth/pair")
    async def pair() -> dict:  # pragma: no cover - raises before returning
        raise RuntimeError("kaboom")

    boom.middleware("http")(rate_limit_middleware)

    client = TestClient(boom, raise_server_exceptions=False)
    for _ in range(PAIR_EXCHANGE_LIMIT * 2):
        client.post("/auth/pair", json={}, headers={"X-Forwarded-For": "203.0.113.12"})
    assert get_limiter().tracked_keys() == 0, "a server crash was charged to the caller"
