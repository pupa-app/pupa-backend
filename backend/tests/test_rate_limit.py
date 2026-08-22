"""Abuse throttling on the pairing endpoints.

`/auth/pair` is the one unauthenticated write route on the backend, so it's
the only surface a stranger can hammer. These cover the limiter itself, the
proxy-aware client key it buckets on, and that it's wired outermost.
"""


from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, rate_limit_middleware, router as auth_router
from pupa_backend.auth.devices import reset_for_testing as reset_devices
from pupa_backend.auth.pairing import reset_for_testing as reset_pairing
from pupa_backend.auth.ratelimit import (
    PAIR_BEGIN_LIMIT,
    PAIR_EXCHANGE_LIMIT,
    SlidingWindowLimiter,
    client_key,
    reset_for_testing as reset_limiter,
)


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.delenv("PUPA_RATE_LIMIT_DISABLED", raising=False)
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


def test_client_key_prefers_the_forwarded_client() -> None:
    """Every transport mode (Tailscale serve, Cloudflare tunnel, Railway)
    terminates in front of a loopback-bound listener, so `client.host` is
    127.0.0.1 for *every* remote caller. Bucketing on it would put the whole
    internet in one bucket."""
    assert client_key(_Req("127.0.0.1", "203.0.113.7")) == "203.0.113.7"


def test_client_key_takes_the_rightmost_forwarded_entry() -> None:
    """A caller can forge `X-Forwarded-For`; the trusted proxy *appends* what
    it actually saw. The rightmost entry is the only one it wrote."""
    assert client_key(_Req("127.0.0.1", "1.1.1.1, 2.2.2.2, 203.0.113.7")) == "203.0.113.7"


def test_client_key_falls_back_to_the_peer_without_a_proxy() -> None:
    assert client_key(_Req("198.51.100.4")) == "198.51.100.4"


def test_client_key_survives_a_missing_peer() -> None:
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
