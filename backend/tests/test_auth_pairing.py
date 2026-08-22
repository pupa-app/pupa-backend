"""Tests for the bootstrap-code pairing flow.

Two layers:
- Unit tests of `PairingCodeStore`: mint/consume semantics, single-use,
  TTL expiry, sweep on every access.
- Integration tests of `/auth/pair/begin` (admin) + `/auth/pair` (public)
  against a minimal FastAPI app that mounts the auth router behind the
  same middleware production uses.
"""


from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, router as auth_router
from pupa_backend.auth.devices import reset_for_testing as reset_devices
from pupa_backend.auth.pairing import DEFAULT_TTL, PairingCodeStore, reset_for_testing as reset_pairing


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    # Fresh device + pairing stores per test so flows don't leak between cases.
    reset_devices(tmp_path / "devices.json")
    reset_pairing()
    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.include_router(auth_router, prefix="/auth")
    return app


# ---------------------------------------------------------------------------
# PairingCodeStore unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mint_returns_unique_codes() -> None:
    store = PairingCodeStore()
    a = await store.mint()
    b = await store.mint()
    assert a.code != b.code
    assert await store.active_count() == 2


@pytest.mark.asyncio
async def test_consume_succeeds_once_then_returns_none() -> None:
    store = PairingCodeStore()
    minted = await store.mint(suggested_label="phone")
    consumed = await store.consume(minted.code)
    assert consumed is not None
    assert consumed.code == minted.code
    assert consumed.suggested_label == "phone"
    # Single-use: a second consume returns None.
    assert await store.consume(minted.code) is None


@pytest.mark.asyncio
async def test_consume_with_unknown_code_returns_none() -> None:
    store = PairingCodeStore()
    assert await store.consume("NOSUCHCODE") is None


@pytest.mark.asyncio
async def test_expired_code_is_unredeemable_and_swept() -> None:
    store = PairingCodeStore()
    # Negative TTL → already expired by the time we try to consume.
    minted = await store.mint(ttl=timedelta(seconds=-1))
    assert await store.consume(minted.code) is None
    # And the sweep dropped it from the store.
    assert await store.active_count() == 0


@pytest.mark.asyncio
async def test_default_ttl_is_five_minutes() -> None:
    # Sanity check on the constant — clients (iOS countdown UI, operator
    # expectation) assume five minutes.
    assert DEFAULT_TTL == timedelta(minutes=5)


# ---------------------------------------------------------------------------
# /auth/pair/begin (admin)
# ---------------------------------------------------------------------------


def test_pair_begin_requires_auth(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post("/auth/pair/begin", json={})
    assert resp.status_code == 401


def test_pair_begin_rejects_a_device_token(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Only the operator mints pairing codes. A device token authenticates
    fine at the middleware but must not be able to mint another device —
    otherwise a leaked token outlives the revocation of its own device.
    """
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin", json={}, headers={"Authorization": "Bearer k"}
    ).json()
    token = client.post(
        "/auth/pair", json={"code": begin["code"], "label": "phone"}
    ).json()["token"]

    resp = client.post(
        "/auth/pair/begin", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 403
    assert "PUPA_API_KEY" in resp.json()["detail"]


def test_pair_begin_rejects_unknown_scopes(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Scope names are a closed set; a typo shouldn't mint a token whose
    scope silently matches nothing."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post(
        "/auth/pair/begin",
        json={"scopes": ["agent", "root"]},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 422
    assert "root" in str(resp.json())


def test_pair_begin_returns_code_and_metadata(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post(
        "/auth/pair/begin",
        json={"label": "iPhone"},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["code"], str) and len(body["code"]) == 8
    assert "expiresAt" in body
    assert body["suggestedLabel"] == "iPhone"
    assert "screenshare" in body["scopes"]


def test_pair_begin_with_custom_scopes(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post(
        "/auth/pair/begin",
        json={"label": "kiosk", "scopes": ["agent"]},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 200
    assert resp.json()["scopes"] == ["agent"]


def test_pair_begin_with_an_empty_scope_list_grants_nothing(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """An explicit `[]` asks for a device with no privileges — the opposite of
    omitting the field. Treating it as falsy would hand the request that most
    clearly asks for least privilege the entire default set, `agent` included.
    """
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post(
        "/auth/pair/begin",
        json={"label": "kiosk", "scopes": []},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 200
    assert resp.json()["scopes"] == []


# ---------------------------------------------------------------------------
# /auth/pair (public exchange)
# ---------------------------------------------------------------------------


def test_pair_exchange_is_public_no_bearer_required(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin",
        json={},
        headers={"Authorization": "Bearer k"},
    ).json()
    # Note: NO Authorization header on /auth/pair — the device doesn't have
    # one yet. The middleware lets this through (`_is_public`).
    resp = client.post("/auth/pair", json={"code": begin["code"], "label": "iPad"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["deviceId"], str) and len(body["deviceId"]) > 0
    assert isinstance(body["token"], str) and len(body["token"]) >= 32
    assert body["label"] == "iPad"
    assert "screenshare" in body["scopes"]


def test_pair_exchange_can_be_used_to_authenticate(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Round-trip: mint a code → redeem it → use the resulting token on a
    gated route. Pins the whole flow end-to-end at the FastAPI layer.
    """
    monkeypatch.setenv("PUPA_API_KEY", "k")

    @app.get("/protected")
    async def protected() -> dict:
        return {"ok": True}

    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin",
        json={},
        headers={"Authorization": "Bearer k"},
    ).json()
    exchange = client.post(
        "/auth/pair", json={"code": begin["code"], "label": "User's iPhone"}
    ).json()
    token = exchange["token"]
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_pair_exchange_is_single_use(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin",
        json={},
        headers={"Authorization": "Bearer k"},
    ).json()
    first = client.post("/auth/pair", json={"code": begin["code"], "label": "iPad"})
    second = client.post("/auth/pair", json={"code": begin["code"], "label": "iPad"})
    assert first.status_code == 200
    assert second.status_code == 404


def test_pair_exchange_with_unknown_code_returns_404(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post("/auth/pair", json={"code": "NOTREAL1", "label": "x"})
    assert resp.status_code == 404


def test_pair_exchange_label_overrides_suggested(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin",
        json={"label": "iPhone (suggested)"},
        headers={"Authorization": "Bearer k"},
    ).json()
    exchange = client.post(
        "/auth/pair", json={"code": begin["code"], "label": "User's iPhone"}
    ).json()
    assert exchange["label"] == "User's iPhone"


# ---------------------------------------------------------------------------
# Per-request TTL knobs (codeTtlSeconds, deviceTokenTtlSeconds) — added for
# cloud deployments where short-lived codes + expiring tokens are required.
# ---------------------------------------------------------------------------


def test_pair_begin_custom_code_ttl_shortens_expiry(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    from datetime import datetime, timezone

    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post(
        "/auth/pair/begin",
        json={"label": "x", "codeTtlSeconds": 10},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 200
    expires_at = datetime.fromisoformat(resp.json()["expiresAt"])
    delta = (expires_at - datetime.now(timezone.utc)).total_seconds()
    # 10s requested; allow generous slack for slow CI.
    assert 0 < delta <= 12


def test_pair_begin_rejects_out_of_range_code_ttl(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.post(
        "/auth/pair/begin",
        json={"codeTtlSeconds": 0},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/auth/pair/begin",
        json={"codeTtlSeconds": 86401},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 422


def test_device_token_expires_when_ttl_set(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Mint a code with a 1-second device-token TTL, redeem it, then sleep
    long enough that the issued token is rejected by the middleware.
    """
    import time

    monkeypatch.setenv("PUPA_API_KEY", "k")

    @app.get("/protected")
    async def protected() -> dict:
        return {"ok": True}

    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin",
        json={"deviceTokenTtlSeconds": 1},
        headers={"Authorization": "Bearer k"},
    ).json()
    exchange = client.post(
        "/auth/pair", json={"code": begin["code"], "label": "phone"}
    ).json()
    token = exchange["token"]
    # Works immediately.
    assert client.get("/protected", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    # Expires after the TTL.
    time.sleep(1.2)
    assert client.get("/protected", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_device_token_without_ttl_never_expires(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Backward-compat: when no `deviceTokenTtlSeconds` is sent, the token
    has no `expiresAt` and the legacy behaviour (no expiry) is preserved.
    """
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin", json={}, headers={"Authorization": "Bearer k"}
    ).json()
    client.post("/auth/pair", json={"code": begin["code"], "label": "phone"})
    devices = client.get("/auth/devices", headers={"Authorization": "Bearer k"}).json()
    assert len(devices) == 1
    assert devices[0]["expiresAt"] is None


def test_devices_list_includes_expires_at(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    begin = client.post(
        "/auth/pair/begin",
        json={"deviceTokenTtlSeconds": 3600},
        headers={"Authorization": "Bearer k"},
    ).json()
    client.post("/auth/pair", json={"code": begin["code"], "label": "phone"})
    devices = client.get("/auth/devices", headers={"Authorization": "Bearer k"}).json()
    assert len(devices) == 1
    assert devices[0]["expiresAt"] is not None
