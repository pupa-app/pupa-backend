"""Per-route authorization tests.

The auth middleware proves *who* a caller is (api_key vs paired device);
``auth.scopes`` enforces *what they can do*. These tests cover each
gated surface against three identities:

- ``api_key`` (operator) — should pass every route.
- ``device with full scopes`` — should pass scope-gated routes but get
  403 on operator-only ones (``/auth/devices/*``).
- ``device missing the relevant scope`` — should get 403 on that route.

``PUPA_AUTH_DISABLED=1`` short-circuits both the middleware and the
scope dependencies; covered explicitly in one case to lock the dev opt-out.
"""


from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, router as auth_router
from pupa_backend.auth.devices import DeviceStore, reset_for_testing
from pupa_backend.harnesses.routes import router as harnesses_router
from pupa_backend.harnesses.langgraph.db.connection import open_persistence
from pupa_backend.harnesses.langgraph.db.routes import router as db_router


@pytest.fixture
def store(tmp_path: Path) -> DeviceStore:
    return reset_for_testing(tmp_path / "devices.json")


@pytest.fixture
async def app(monkeypatch: pytest.MonkeyPatch):
    """FastAPI app with auth + db + harnesses mounted and a live in-memory
    checkpointer, held open for the duration of the test so /db/* handlers
    see an initialised saver."""
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.include_router(auth_router, prefix="/auth")
    app.include_router(db_router, prefix="/db")
    app.include_router(harnesses_router, prefix="/harnesses")
    async with open_persistence(None, None) as (checkpointer, _store):
        app.state.checkpointer = checkpointer
        yield app


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _device_token(
    store: DeviceStore, scopes: list[str] | None = None
) -> str:
    label = "device-" + ",".join(scopes or ["default"])
    if scopes is None:
        _id, token = await store.issue(label=label)
    else:
        _id, token = await store.issue(label=label, scopes=scopes)
    return token


# ---------------------------------------------------------------------------
# /harnesses — `agent` scope (harness discovery: models + tools + permissions)
# ---------------------------------------------------------------------------


async def test_harnesses_accepts_api_key(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    assert client.get("/harnesses", headers=_bearer("k")).status_code == 200


@pytest.mark.asyncio
async def test_harnesses_accepts_device_with_agent_scope(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    token = await _device_token(store, ["agent"])
    client = TestClient(app)
    assert client.get("/harnesses", headers=_bearer(token)).status_code == 200


@pytest.mark.asyncio
async def test_harnesses_rejects_device_without_agent_scope(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    token = await _device_token(store, ["tools", "db", "memory"])  # no "agent"
    client = TestClient(app)
    resp = client.get("/harnesses", headers=_bearer(token))
    assert resp.status_code == 403
    assert "agent" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /db/threads/* — `agent` scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_threads_accepts_device_with_agent_scope(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    token = await _device_token(store, ["agent"])
    client = TestClient(app)
    # No data → 200 with [] (per get_thread_messages semantics).
    resp = client.get("/db/threads/some-thread/messages", headers=_bearer(token))
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_threads_rejects_device_without_agent_scope(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    token = await _device_token(store, ["tools", "memory"])  # no "agent"
    client = TestClient(app)
    resp = client.get("/db/threads/some-thread/messages", headers=_bearer(token))
    assert resp.status_code == 403
    assert "agent" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /auth/devices — operator-only
# ---------------------------------------------------------------------------


async def test_operator_route_accepts_api_key(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """The operator key reaches an operator-only route; 403 would mean we
    blocked the one identity that is supposed to pass."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    client = TestClient(app)
    resp = client.get("/auth/devices", headers=_bearer("k"))
    assert resp.status_code == 200


async def test_operator_route_rejects_device_even_with_all_scopes(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    """Operator-only means operator-only: a fully-scoped device is still
    blocked, and the 403 says why."""
    monkeypatch.setenv("PUPA_API_KEY", "k")
    token = await _device_token(store)  # DEFAULT_SCOPES
    client = TestClient(app)
    resp = client.get("/auth/devices", headers=_bearer(token))
    assert resp.status_code == 403
    assert "operator-only" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_revoke_device_rejects_device_bearer(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "k")
    other_id, _ = await store.issue(label="iphone")
    attacker = await _device_token(store)
    client = TestClient(app)
    # A paired device tries to revoke a sibling — must be blocked.
    resp = client.delete(f"/auth/devices/{other_id}", headers=_bearer(attacker))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Post-bootstrap state: operator has unset PUPA_API_KEY after pairing.
# Operator-only routes must stay locked — a device bearer should still 403,
# and the routes are effectively unreachable until the key is re-set.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_devices_rejects_device_with_no_api_key_configured(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, store: DeviceStore
) -> None:
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    token = await _device_token(store)
    client = TestClient(app)
    resp = client.get("/auth/devices", headers=_bearer(token))
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_operator_only_routes_unreachable_without_api_key_or_token(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """No `PUPA_API_KEY`, no paired devices, no bearer header — the
    middleware fires first and returns 401 before our scope deps run."""
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    client = TestClient(app)
    assert client.get("/auth/devices").status_code == 401


# ---------------------------------------------------------------------------
# PUPA_AUTH_DISABLED — same-laptop dev opt-out bypasses scope checks too
# ---------------------------------------------------------------------------


async def test_auth_disabled_bypasses_all_scopes(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")
    client = TestClient(app)
    assert client.get("/harnesses").status_code == 200
    assert client.get("/db/threads/x/messages").status_code == 200
    assert client.get("/auth/devices").status_code == 200
