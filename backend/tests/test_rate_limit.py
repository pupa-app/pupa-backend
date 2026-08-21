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


def test_limiter_allows_up_to_the_limit_then_blocks() -> None:
    clock = iter([0.0] * 10)
    limiter = SlidingWindowLimiter(now=lambda: next(clock))
    assert [limiter.allow("ip", limit=3, window=60.0) for _ in range(4)] == [
        True, True, True, False,
    ]


def test_limiter_forgets_hits_older_than_the_window() -> None:
    times = [0.0, 1.0, 2.0, 61.0]
    clock = iter(times)
    limiter = SlidingWindowLimiter(now=lambda: next(clock))
    assert limiter.allow("ip", limit=3, window=60.0)
    assert limiter.allow("ip", limit=3, window=60.0)
    assert limiter.allow("ip", limit=3, window=60.0)
    # t=61 — the first three hits have aged out of the 60 s window.
    assert limiter.allow("ip", limit=3, window=60.0)


def test_limiter_buckets_are_independent_per_key() -> None:
    limiter = SlidingWindowLimiter(now=lambda: 0.0)
    assert limiter.allow("a", limit=1, window=60.0)
    assert not limiter.allow("a", limit=1, window=60.0)
    assert limiter.allow("b", limit=1, window=60.0)


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
# /auth/pair throttling
# ---------------------------------------------------------------------------


def _hammer(client: TestClient, n: int, xff: str = "203.0.113.7") -> list[int]:
    return [
        client.post(
            "/auth/pair",
            json={"code": "NOSUCHCO", "label": "x"},
            headers={"X-Forwarded-For": xff},
        ).status_code
        for _ in range(n)
    ]


def test_pair_exchange_throttles_after_the_limit(app: FastAPI) -> None:
    client = TestClient(app)
    codes = _hammer(client, PAIR_EXCHANGE_LIMIT + 1)
    # Wrong codes, so every allowed attempt is a 404 — the point is the last one.
    assert codes[:-1] == [404] * PAIR_EXCHANGE_LIMIT
    assert codes[-1] == 429


def test_throttling_is_per_client(app: FastAPI) -> None:
    client = TestClient(app)
    assert _hammer(client, PAIR_EXCHANGE_LIMIT)[-1] == 404
    # A different forwarded client gets its own bucket.
    assert _hammer(client, 1, xff="198.51.100.9") == [404]


def test_rate_limit_can_be_disabled_for_dev(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUPA_RATE_LIMIT_DISABLED", "1")
    client = TestClient(app)
    assert 429 not in _hammer(client, PAIR_EXCHANGE_LIMIT + 3)


def test_throttled_response_says_how_long_to_wait(app: FastAPI) -> None:
    client = TestClient(app)
    codes = _hammer(client, PAIR_EXCHANGE_LIMIT + 1)
    assert codes[-1] == 429
    resp = client.post(
        "/auth/pair",
        json={"code": "NOSUCHCO", "label": "x"},
        headers={"X-Forwarded-For": "203.0.113.7"},
    )
    assert resp.headers.get("retry-after") is not None


def test_unrelated_routes_are_not_throttled(app: FastAPI) -> None:
    client = TestClient(app)
    for _ in range(PAIR_EXCHANGE_LIMIT + 5):
        assert client.get("/auth/config").status_code == 200

