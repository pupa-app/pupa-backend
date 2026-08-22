"""`PUPA_REQUIRE_HTTPS` — opt-in refusal to serve plaintext.

TLS stays optional by design (offline/LAN self-hosts), so this is a flag the
operator sets on anything internet-reachable. The pairing handshake is the
reason it matters: the device token is handed over in the clear exactly once,
and on a plaintext hop anyone on the path gets it.
"""


from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware, require_https_middleware, router as auth_router
from pupa_backend.auth.devices import reset_for_testing as reset_devices
from pupa_backend.auth.pairing import reset_for_testing as reset_pairing
from pupa_backend.auth.transport import is_secure_request


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.delenv("PUPA_REQUIRE_HTTPS", raising=False)
    reset_devices(tmp_path / "devices.json")
    reset_pairing()
    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(require_https_middleware)
    app.include_router(auth_router, prefix="/auth")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    return app


def _pair(client: TestClient, **headers: str):
    return client.post("/auth/pair", json={"code": "NOSUCHCO", "label": "x"}, headers=headers)


# ---------------------------------------------------------------------------
# is_secure_request
# ---------------------------------------------------------------------------


class _Req:
    def __init__(self, scheme: str, proto: str | None = None) -> None:
        self.url = type("U", (), {"scheme": scheme})()
        self.headers = {"x-forwarded-proto": proto} if proto else {}


@pytest.fixture(autouse=True)
def _trusted_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most cases here model a proxied deployment. The untrusted case — where
    the header is just a string the caller chose — is pinned separately."""
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")


def test_direct_tls_is_secure() -> None:
    assert is_secure_request(_Req("https"))


def test_forwarded_proto_is_secure() -> None:
    """Every tunnel mode terminates TLS and forwards over a loopback hop, so
    the scheme this process sees is `http` even when the caller used HTTPS.
    The proxy's `X-Forwarded-Proto` is the only signal of the real hop."""
    assert is_secure_request(_Req("http", "https"))


def test_forwarded_proto_takes_the_rightmost_entry() -> None:
    assert is_secure_request(_Req("http", "http, https"))
    assert not is_secure_request(_Req("http", "https, http"))


def test_plaintext_is_not_secure() -> None:
    assert not is_secure_request(_Req("http"))
    assert not is_secure_request(_Req("http", "http"))


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def test_plaintext_pairing_is_refused_when_required(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    resp = _pair(TestClient(app))
    assert resp.status_code == 403
    assert "HTTPS" in resp.json()["detail"]


def test_forwarded_https_pairing_is_allowed_when_required(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    # 404 = it reached the handler and the bogus code missed, i.e. not blocked.
    assert _pair(TestClient(app), **{"X-Forwarded-Proto": "https"}).status_code == 404


def test_plaintext_pairing_is_allowed_when_the_flag_is_unset(app: FastAPI) -> None:
    """LAN and offline self-hosts are the default case and must keep working —
    that's why this is opt-in rather than always-on."""
    assert _pair(TestClient(app)).status_code == 404


def test_health_stays_reachable_over_plaintext(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Platform health probes hit the listener directly, behind the TLS
    terminator — blocking them would fail the deploy the flag is meant for."""
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    assert TestClient(app).get("/health").status_code == 200


def test_the_flag_covers_every_other_route_too(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just pairing: a plaintext hop leaks the bearer token on any request
    that carries one."""
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    assert TestClient(app).get("/auth/config").status_code == 403


def test_loopback_gets_no_exemption(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is deliberately no 'but it came from 127.0.0.1' carve-out. Every
    transport mode terminates on a loopback-bound listener, so an IP-based
    exemption would exempt the entire internet. Local dev leaves the flag off
    instead."""
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    resp = _pair(TestClient(app, base_url="http://127.0.0.1"))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Screen-share signalling socket
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    from pupa_backend.screenshare import router as screenshare_router

    monkeypatch.delenv("PUPA_REQUIRE_HTTPS", raising=False)
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    reset_devices(tmp_path / "devices.json")
    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(require_https_middleware)
    app.include_router(screenshare_router, prefix="/screenshare")
    return app


@pytest.mark.asyncio
async def test_plaintext_screenshare_socket_is_refused_when_required(
    ws_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP middleware stack never runs for a WebSocket, so this needs its
    own check — and it needs one: the signalling socket sends the device token
    in an Authorization header, so a `ws://` hop leaks it the same way."""
    from pupa_backend.auth.devices import get_store

    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    _id, token = await get_store().issue(label="phone", scopes=["screenshare"])
    client = TestClient(ws_app)
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(
            "/screenshare/ws?role=publisher&share_id=any",
            headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            ws.receive_json()
    assert getattr(excinfo.value, "code", None) == 4403


@pytest.mark.asyncio
async def test_forwarded_https_screenshare_socket_is_allowed(
    ws_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pupa_backend.auth.devices import get_store

    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    _id, token = await get_store().issue(label="phone", scopes=["screenshare"])
    client = TestClient(ws_app)
    with client.websocket_connect(
        "/screenshare/ws?role=publisher&share_id=any",
        headers={"Authorization": f"Bearer {token}", "X-Forwarded-Proto": "https"},
    ) as ws:
        assert ws is not None


def test_forwarded_proto_is_ignored_without_a_proxy_in_front(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise a client on a plaintext hop asserts its own hop was TLS and
    walks straight through PUPA_REQUIRE_HTTPS — the flag would be decoration."""
    monkeypatch.delenv("PUPA_TRUSTED_PROXY", raising=False)
    monkeypatch.delenv("PUPA_CONNECTIVITY", raising=False)
    assert not is_secure_request(_Req("http", "https"))


def test_a_forged_proto_cannot_pass_the_middleware(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUPA_REQUIRE_HTTPS", "1")
    monkeypatch.delenv("PUPA_TRUSTED_PROXY", raising=False)
    monkeypatch.delenv("PUPA_CONNECTIVITY", raising=False)
    resp = _pair(TestClient(app), **{"X-Forwarded-Proto": "https"})
    assert resp.status_code == 403


def test_tailscale_and_cloudflared_are_trusted_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This process starts those proxies itself, so it knows one is in front —
    operators shouldn't have to set a second flag to keep working deployments
    working."""
    monkeypatch.delenv("PUPA_TRUSTED_PROXY", raising=False)
    for mode in ("tailscale", "cloudflared"):
        monkeypatch.setenv("PUPA_CONNECTIVITY", mode)
        assert is_secure_request(_Req("http", "https")), mode


def test_an_explicit_flag_overrides_the_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "0")
    assert not is_secure_request(_Req("http", "https"))

# ---------------------------------------------------------------------------
# Forwarded headers: every line, not just the first
# ---------------------------------------------------------------------------


def test_a_second_forwarded_line_cannot_be_hidden_behind_the_callers_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy may *append a field line* rather than extend the caller's (Go's
    `Header.Add`, HAProxy's `option forwardfor`). Reading only the first line
    would let the caller send `X-Forwarded-Proto: https`, have the proxy add
    its real `http` underneath, and still be believed."""
    from starlette.datastructures import Headers

    from pupa_backend.auth.proxy import forwarded_values

    headers = Headers(
        raw=[
            (b"x-forwarded-proto", b"https"),   # written by the caller
            (b"x-forwarded-proto", b"http"),    # appended by the proxy
        ]
    )
    assert forwarded_values(headers, "x-forwarded-proto") == ["https", "http"]


async def test_a_forged_proto_line_cannot_outrank_the_proxys_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same case, end to end: the rightmost hop is the proxy's, so a plaintext
    hop stays refused however the caller arranges its own headers."""
    from starlette.requests import Request

    monkeypatch.setenv("PUPA_TRUSTED_PROXY", "1")
    scope = {
        "type": "http",
        "scheme": "http",
        "method": "POST",
        "path": "/auth/pair",
        "query_string": b"",
        "headers": [
            (b"x-forwarded-proto", b"https"),
            (b"x-forwarded-proto", b"http"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    assert not is_secure_request(Request(scope))
