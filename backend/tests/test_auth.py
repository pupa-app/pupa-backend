"""Tests for the auth middleware + `/auth/config` probe.

Auth is required **by default**. The only way to disable
it is `PUPA_AUTH_DISABLED=1`. `PUPA_API_KEY` is the server-side
bootstrap credential used by `make pair` and as an alternative bearer.

Mounts the middleware on a minimal test FastAPI app rather than the
production `app.py` — the latter imports `agent.py` which builds a
Bedrock/Anthropic client at module load. The middleware doesn't depend on
the agent.
"""



import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, router as auth_router


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    # Each test starts from a clean env — the middleware default is
    # "auth required" so individual tests opt in to a different state.
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)

    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.include_router(auth_router, prefix="/auth")

    @app.get("/protected")
    async def protected() -> dict:
        return {"ok": True}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


def test_default_requires_auth_and_rejects_no_bearer(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """With no env vars set at all, the middleware requires a Bearer header.
    No paired device, no api key, no header → 401."""
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].startswith("Bearer")


def test_disabled_env_opens_all_routes(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Explicit `PUPA_AUTH_DISABLED=1` opt-out — used for the
    same-laptop `make backend` + `make mac-demo` dev flow."""
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")
    client = TestClient(app)
    assert client.get("/protected").status_code == 200
    assert client.get("/auth/config").json()["authRequired"] is False


def test_api_key_set_accepts_correct_bearer(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer s3cr3t"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_api_key_set_rejects_wrong_token(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_api_key_set_rejects_non_bearer_scheme(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Basic s3cr3t"})
    assert resp.status_code == 401


def test_auth_config_default_required_no_methods(app: FastAPI) -> None:
    """Default state: auth required, no `api_key` method advertised because
    no `PUPA_API_KEY` is set. iOS clients see this as 🔒 "needs pairing"."""
    client = TestClient(app)
    body = client.get("/auth/config").json()
    assert body["authRequired"] is True
    assert body["methods"] == []


def test_auth_config_required_with_api_key_method(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    body = client.get("/auth/config").json()
    assert body["authRequired"] is True
    assert body["methods"] == ["api_key"]
    assert "version" in body


def test_auth_config_disabled_reports_false(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")  # ignored when disabled
    client = TestClient(app)
    body = client.get("/auth/config").json()
    assert body["authRequired"] is False
    assert body["methods"] == []


@pytest.mark.parametrize("value", ["0", "false", "False", "no", ""])
def test_auth_disabled_env_falsy_keeps_auth_required(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, value: str
) -> None:
    """Falsy values for `PUPA_AUTH_DISABLED` are treated as off — i.e.
    auth stays required. Same truthy-table as `PUPA_SCREENSHARE`."""
    monkeypatch.setenv("PUPA_AUTH_DISABLED", value)
    client = TestClient(app)
    assert client.get("/auth/config").json()["authRequired"] is True
    # And the gated route still requires a Bearer.
    assert client.get("/protected").status_code == 401


def test_auth_config_always_public(app: FastAPI) -> None:
    """/auth/config is reachable even when auth is required and the caller
    has no credential — the iOS Settings probe relies on this."""
    client = TestClient(app)
    resp = client.get("/auth/config")
    assert resp.status_code == 200


def test_health_always_public(app: FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
