"""Tests for the paired-device token store + its integration with the auth
middleware and the screen-share broker auth.

The store is JSON-backed; tests use a fresh temp path per case via
`reset_for_testing(path)` so they don't share state.
"""


import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, router as auth_router
from pupa_backend.auth.devices import DEFAULT_SCOPES, DeviceStore, get_store, reset_for_testing
from pupa_backend.screenshare import router as screenshare_router


@pytest.fixture
def store(tmp_path: Path) -> DeviceStore:
    """Fresh DeviceStore singleton backed by a tmp file per test."""
    return reset_for_testing(tmp_path / "devices.json")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    # Default-required auth. Tests opt in to specific
    # states by setting env vars themselves.
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)

    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.include_router(auth_router, prefix="/auth")
    app.include_router(screenshare_router, prefix="/screenshare")

    @app.get("/protected")
    async def protected() -> dict:
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# DeviceStore unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_returns_id_and_plaintext_token(store: DeviceStore) -> None:
    device_id, token = await store.issue(label="phone", scopes=["agent", "screenshare"])
    assert isinstance(device_id, str) and len(device_id) > 0
    assert isinstance(token, str) and len(token) >= 32
    # The plaintext token is NEVER stored on disk — only its hash.
    on_disk = json.loads((store._path).read_text())
    for entry in on_disk["devices"].values():
        assert entry.get("token") is None
        assert token not in json.dumps(entry)


@pytest.mark.asyncio
async def test_resolve_returns_device_for_active_token(store: DeviceStore) -> None:
    _id, token = await store.issue(label="phone")
    device = await store.resolve(token)
    assert device is not None
    assert device.label == "phone"
    assert device.scopes == DEFAULT_SCOPES


@pytest.mark.asyncio
async def test_resolve_returns_none_for_unknown_token(store: DeviceStore) -> None:
    assert await store.resolve("not-a-real-token") is None


@pytest.mark.asyncio
async def test_revoke_makes_token_unresolvable(store: DeviceStore) -> None:
    device_id, token = await store.issue(label="phone")
    assert await store.revoke(device_id) is True
    assert await store.resolve(token) is None
    # Second revoke is a no-op — returns False.
    assert await store.revoke(device_id) is False


@pytest.mark.asyncio
async def test_list_active_hides_revoked(store: DeviceStore) -> None:
    a_id, _ = await store.issue(label="iphone")
    b_id, _ = await store.issue(label="ipad")
    await store.revoke(a_id)
    active = await store.list_active()
    assert [d.id for d in active] == [b_id]


@pytest.mark.asyncio
async def test_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "devices.json"
    writer = DeviceStore(path)
    _id, token = await writer.issue(label="phone")
    reader = DeviceStore(path)
    device = await reader.resolve(token)
    assert device is not None
    assert device.label == "phone"


def test_default_store_path_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the default device-store path must be absolute and anchored
    to ~/.pupa-backend, never CWD-relative. The `pupa-backend` console script
    runs from whatever directory it's invoked in; a CWD-relative default made
    each launch dir read a different (empty) store and silently unpair every
    device."""
    from pupa_backend.auth.devices import DEFAULT_PATH

    expected = Path.home() / ".pupa-backend" / "pupa-auth.json"
    assert DEFAULT_PATH == expected
    assert DEFAULT_PATH.is_absolute()
    # Same path regardless of the working directory the process started in.
    monkeypatch.chdir(tmp_path)
    assert DeviceStore()._path == expected


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_accepts_device_token(
    app: FastAPI, store: DeviceStore
) -> None:
    """A paired device's token unlocks the protected route even with no
    `PUPA_API_KEY` set — auth is required by default; the device
    store is the only source of valid tokens until the operator bootstraps."""
    _id, token = await store.issue(label="phone")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_middleware_rejects_revoked_device(
    app: FastAPI, store: DeviceStore
) -> None:
    device_id, token = await store.issue(label="phone")
    await store.revoke(device_id)
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_middleware_accepts_bootstrap_api_key(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore  # noqa: ARG001
) -> None:
    """A request bearing the server-side `PUPA_API_KEY` is accepted —
    same bootstrap credential `make pair` uses against `/auth/pair/begin`."""
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer s3cr3t"})
    assert resp.status_code == 200


def test_middleware_rejects_random_token(
    app: FastAPI, store: DeviceStore  # noqa: ARG001
) -> None:
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer junk"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /auth/devices routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_devices_returns_active_only_without_tokens(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    a_id, _ = await store.issue(label="iphone")
    b_id, _ = await store.issue(label="ipad")
    await store.revoke(a_id)
    client = TestClient(app)
    resp = client.get("/auth/devices", headers={"Authorization": "Bearer k"})
    assert resp.status_code == 200
    body = resp.json()
    assert [d["id"] for d in body] == [b_id]
    # No token / hash in the response.
    for d in body:
        assert "token" not in d
        assert "tokenHash" not in d


@pytest.mark.asyncio
async def test_delete_device_revokes_then_returns_404_on_repeat(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    device_id, _ = await store.issue(label="phone")
    client = TestClient(app)
    first = client.delete(f"/auth/devices/{device_id}", headers={"Authorization": "Bearer k"})
    assert first.status_code == 200
    assert first.json() == {"revoked": device_id}

    second = client.delete(f"/auth/devices/{device_id}", headers={"Authorization": "Bearer k"})
    assert second.status_code == 404


def test_devices_routes_require_auth(
    app: FastAPI, store: DeviceStore  # noqa: ARG001
) -> None:
    client = TestClient(app)
    assert client.get("/auth/devices").status_code == 401
    assert client.delete("/auth/devices/any-id").status_code == 401


# ---------------------------------------------------------------------------
# Screen-share broker auth integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_screenshare_ws_accepts_device_with_screenshare_scope(
    app: FastAPI, store: DeviceStore
) -> None:
    _id, token = await store.issue(label="phone", scopes=["agent", "screenshare"])
    client = TestClient(app)
    # publisher first so the viewer can pair.
    with client.websocket_connect(
        "/screenshare/ws?role=publisher&share_id=share-1",
        headers={"Authorization": f"Bearer {token}"},
    ) as pub:
        with client.websocket_connect(
            "/screenshare/ws?role=viewer&share_id=share-1",
            headers={"Authorization": f"Bearer {token}"},
        ) as viewer:
            assert pub.receive_json() == {"type": "viewer_joined"}
            pub.send_json({"type": "offer", "sdp": "fake"})
            assert viewer.receive_json() == {"type": "offer", "sdp": "fake"}


@pytest.mark.asyncio
async def test_screenshare_ws_rejects_device_without_screenshare_scope(
    app: FastAPI, store: DeviceStore
) -> None:
    _id, token = await store.issue(label="phone", scopes=["agent"])  # no screenshare
    client = TestClient(app)
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(
            "/screenshare/ws?role=publisher&share_id=any",
            headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            ws.receive_json()
    assert getattr(excinfo.value, "code", None) == 4401
