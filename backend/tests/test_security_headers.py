"""Response hardening headers."""


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import security_headers_middleware


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.delenv("PUPA_REQUIRE_HTTPS", raising=False)
    monkeypatch.delenv("PUPA_TLS_CERT", raising=False)
    monkeypatch.delenv("PUPA_TLS_KEY", raising=False)
    # HSTS keys off `is_secure_request`, which only reads X-Forwarded-Proto
    # where a proxy is trusted. Pin both inputs — inheriting either from the
    # ambient environment makes these pass or fail by run order.
    monkeypatch.delenv("PUPA_CONNECTIVITY", raising=False)
    monkeypatch.delenv("PUPA_TRUSTED_PROXY", raising=False)
    app = FastAPI()
    app.middleware("http")(security_headers_middleware)

    @app.get("/x")
    async def x() -> dict:
        return {"ok": True}

    return app


def test_static_headers_are_present(app: FastAPI) -> None:
    headers = TestClient(app).get("/x").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_no_hsts_over_plaintext(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asserting HSTS on a plaintext hop says nothing, and pinning a LAN host
    to HTTPS it doesn't serve would lock the operator out for the max-age."""
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    assert "strict-transport-security" not in TestClient(app).get("/x").headers


def test_hsts_on_a_tunnel_tls_hop_without_a_local_cert(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tunnel modes are the internet-reachable deployments and the ones
    with no local cert: tailscaled / cloudflared hold it and speak plain HTTP
    over loopback. HSTS keys off whether the *hop* was TLS, not off whether
    this process happens to own the certificate."""
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")
    resp = TestClient(app).get("/x", headers={"X-Forwarded-Proto": "https"})
    assert resp.headers["strict-transport-security"].startswith("max-age=")


def test_hsts_on_a_tls_hop_when_https_is_required(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")
    resp = TestClient(app).get("/x", headers={"X-Forwarded-Proto": "https"})
    assert resp.headers["strict-transport-security"].startswith("max-age=")


def test_no_hsts_from_a_forged_proto_header(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a trusted proxy the header is caller-written, so it can't be
    what makes the backend assert HSTS — that would let a stranger pin a LAN
    host to HTTPS it doesn't serve for the lifetime of the max-age."""
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    resp = TestClient(app).get("/x", headers={"X-Forwarded-Proto": "https"})
    assert "strict-transport-security" not in resp.headers


def test_no_cors_headers_are_added(app: FastAPI) -> None:
    """There is no browser origin to allow, and a permissive CORS policy would
    let any web page call this backend with a user's credentials."""
    resp = TestClient(app).get("/x", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
