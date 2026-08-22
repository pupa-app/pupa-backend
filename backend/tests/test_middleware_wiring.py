"""Pins on the production middleware stack.

The other auth tests build their own app and replicate `app.py`'s middleware
order by hand, which means they'd keep passing if the real app stopped
mounting a guard. These read the real stack.

Order is the whole point of each assertion:
- the limiter must be **outermost** — it protects a pre-auth route, so it has
  to run before auth and regardless of the outcome;
- the run-scope guard must be **inside** auth — it reads the identity auth
  resolves, so outside it there'd be nothing to read and every run request
  would 401.
"""


import os
from typing import Any

import pytest

from pupa_backend.auth import (
    api_key_middleware,
    rate_limit_middleware,
    require_https_middleware,
    run_scope_middleware,
    security_headers_middleware,
)


@pytest.fixture
def dispatch_stack() -> list[Any]:
    """Middleware dispatch callables, outermost first.

    Importing `pupa_backend.app` has a global side effect — with the Claude
    harness enabled it moves billing credentials out of `os.environ` at import
    time (`credentials.stash_forbidden_credentials`). Snapshot and restore, or
    every later test that expects those vars fails depending on run order.
    """
    snapshot = dict(os.environ)
    try:
        from pupa_backend.app import app
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
    return [m.kwargs["dispatch"] for m in app.user_middleware if "dispatch" in m.kwargs]


def test_headers_wrap_everything(dispatch_stack: list[Any]) -> None:
    """Outermost, so the guards' own 403/429 responses carry the headers too —
    those never reach the inner stack."""
    assert dispatch_stack, "no http middleware mounted at all"
    assert dispatch_stack[0] is security_headers_middleware


def test_the_https_guard_is_the_outermost_guard(dispatch_stack: list[Any]) -> None:
    """First thing a request hits, after the header pass — and specifically
    *outside* the limiter. A plaintext hop is a misconfiguration, not a guess
    at a credential, so refusing it before the limiter sees it is what keeps it
    from spending a real device's pairing budget. Anything else needs the
    limiter to know about transport, which is a coupling this ordering buys
    its way out of."""
    guards = [m for m in dispatch_stack if m is not security_headers_middleware]
    assert guards[0] is require_https_middleware


def test_run_scope_guard_is_mounted_inside_auth(dispatch_stack: list[Any]) -> None:
    assert run_scope_middleware in dispatch_stack, "run-scope guard is not mounted"
    assert api_key_middleware in dispatch_stack
    assert dispatch_stack.index(api_key_middleware) < dispatch_stack.index(run_scope_middleware)


def test_limiter_is_mounted_outside_auth(dispatch_stack: list[Any]) -> None:
    """It protects a pre-auth route, so it has to run before auth and
    regardless of the outcome."""
    assert rate_limit_middleware in dispatch_stack, "limiter is not mounted"
    assert dispatch_stack.index(require_https_middleware) < dispatch_stack.index(rate_limit_middleware)
    assert dispatch_stack.index(rate_limit_middleware) < dispatch_stack.index(api_key_middleware)


# ---------------------------------------------------------------------------
# The guard's data source, not just its presence
# ---------------------------------------------------------------------------


def test_lifespan_records_the_run_paths_the_guard_reads(tmp_path) -> None:
    """`run_scope_middleware` gates on `app.state.run_paths`, which only exists
    because lifespan startup fills it in. Mounting the middleware without that
    assignment un-gates `POST /` completely while every other test stays green
    — so this one runs the real lifespan and checks the data is there.
    """
    import os

    from fastapi.testclient import TestClient

    snapshot = dict(os.environ)
    try:
        # The real lifespan has real side effects, and it reads the *ambient*
        # environment. Pin every var that would reach outside this test before
        # entering it: a developer with `DATABASE_URL` exported would otherwise
        # run the checkpointer DDL against their live Postgres, `cloudflared`
        # would be spawned as a child, configured MCP servers would be started,
        # and the screenshare token of a backend they already have running
        # would be regenerated on entry and revoked on exit.
        for var in (
            "PUPA_CONNECTIVITY",
            "PUPA_MCP_SERVERS",
            "PUPA_REQUIRE_DB_SCHEME",
        ):
            os.environ.pop(var, None)
        os.environ["PUPA_SCREENSHARE"] = "0"
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'lifespan.db'}"

        from pupa_backend.app import app

        # TestClient's context manager runs startup/shutdown for real.
        with TestClient(app):
            run_paths = getattr(app.state, "run_paths", None)
            assert run_paths, "lifespan did not record any run paths"
            assert "/" in run_paths, (
                "the default harness alias POST / is not gated — a token "
                "without the `agent` scope could run the agent"
            )
            registry = getattr(app.state, "harness_registry", None)
            if registry is not None:
                for harness in registry.enabled():
                    assert f"/harnesses/{harness.id}" in run_paths, harness.id
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_uvicorn_does_not_rewrite_the_scope_from_forwarded_headers(
    monkeypatch,
) -> None:
    """`auth/proxy.py` is the only thing allowed to decide whether
    `X-Forwarded-*` is believable. uvicorn's own `proxy_headers` default folds
    those headers into `scope["scheme"]` and `scope["client"]` for any peer in
    `forwarded_allow_ips` — `127.0.0.1`, which is every tunnel deployment and
    anything else sharing the host. That runs *above* the app, so
    `is_secure_request` would read a forged scheme before ever consulting
    `trust_forwarded_headers`, and the rate limiter would bucket on a forged
    peer. Must stay off.
    """
    import uvicorn

    from pupa_backend import app as app_module

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.delenv("PUPA_CONNECTIVITY", raising=False)
    monkeypatch.delenv("PUPA_TLS_CERT", raising=False)
    monkeypatch.delenv("PUPA_TLS_KEY", raising=False)

    app_module.main()

    assert captured.get("proxy_headers") is False


def _run_main(monkeypatch, **env) -> dict:
    """Run `app.main()` with uvicorn stubbed; return the kwargs it was given."""
    import uvicorn

    from pupa_backend import app as app_module

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr(app_module, "_start_cloudflared_tunnel", lambda: None)
    for var in ("PUPA_HOST", "PUPA_TLS_CERT", "PUPA_TLS_KEY", "PUPA_CONNECTIVITY",
                "PUPA_TRUSTED_PROXY"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app_module.main()
    return captured


def test_tailscale_without_an_active_serve_keeps_the_documented_wildcard_bind(
    monkeypatch,
) -> None:
    """`start_serve_proxy` returns None when the CLI is missing, when
    `PUPA_TAILSCALE_SERVE=0`, or when `serve` fails — all documented as falling
    back to `0.0.0.0`. Binding loopback on the *intent* alone would take a
    working LAN deployment off the network with nothing in the log to explain
    it. And with nothing in front, the forwarded headers are caller-written.
    """
    import os

    from pupa_backend import tailscale_proxy

    monkeypatch.setattr(tailscale_proxy, "start_serve_proxy", lambda port: None)
    captured = _run_main(monkeypatch, PUPA_CONNECTIVITY="tailscale")

    assert captured.get("host") == "0.0.0.0"
    assert os.environ["PUPA_TRUSTED_PROXY"] == "0"


def test_tailscales_raw_tcp_mode_is_not_trusted_to_write_forwarded_headers(
    monkeypatch,
) -> None:
    """`tcp` mode is an L4 passthrough: the client's request arrives
    byte-for-byte, so `X-Forwarded-*` is whatever the caller sent. The listener
    still binds loopback — tailscaled is what reaches it — but nothing it
    forwards may be believed."""
    import os

    from pupa_backend import tailscale_proxy

    class _TcpProxy:
        mode = "tcp"
        terminates_tls = False

        def stop(self) -> None:
            pass

    monkeypatch.setattr(tailscale_proxy, "start_serve_proxy", lambda port: _TcpProxy())
    captured = _run_main(monkeypatch, PUPA_CONNECTIVITY="tailscale")

    assert captured.get("host") == "127.0.0.1"
    assert os.environ["PUPA_TRUSTED_PROXY"] == "0"


def test_an_explicit_operator_setting_still_wins_at_startup(monkeypatch) -> None:
    """`main()` records what it observed, but the operator's answer is the one
    `auth/proxy.py` documents as final."""
    import os

    from pupa_backend import tailscale_proxy

    monkeypatch.setattr(tailscale_proxy, "start_serve_proxy", lambda port: None)
    _run_main(monkeypatch, PUPA_CONNECTIVITY="tailscale", PUPA_TRUSTED_PROXY="1")

    assert os.environ["PUPA_TRUSTED_PROXY"] == "1"


def test_an_inferred_proxy_deployment_binds_loopback(monkeypatch) -> None:
    """`auth/proxy.py` infers forwarded-header trust from `PUPA_CONNECTIVITY`.
    Wherever it does, the listener must not also be reachable directly — a
    wildcard bind would let anyone who can route to the port write their own
    `X-Forwarded-Proto`/`-For` and walk through the HTTPS check and the rate
    limiter. `cloudflared` is the case that used to bind `0.0.0.0`: unlike
    Tailscale, nothing in this process forwards for it.
    """
    import uvicorn

    from pupa_backend import app as app_module

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setenv("PUPA_CONNECTIVITY", "cloudflared")
    monkeypatch.delenv("PUPA_HOST", raising=False)
    monkeypatch.delenv("PUPA_TLS_CERT", raising=False)
    monkeypatch.delenv("PUPA_TLS_KEY", raising=False)
    # Don't actually start a tunnel.
    monkeypatch.setattr(app_module, "_start_cloudflared_tunnel", lambda: None)

    app_module.main()

    assert captured.get("host") == "127.0.0.1"


def test_the_guard_fails_closed_without_run_paths(monkeypatch) -> None:
    """If the attribute is missing the guard must refuse, not wave the request
    through — that branch is the difference between a bug and an open door.

    The caller here holds a *valid* operator key on purpose: with a bad one
    `api_key_middleware` answers 401 first and the fail-closed branch is never
    reached, so the test would pass no matter what this guard does.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pupa_backend.auth import api_key_middleware

    monkeypatch.setenv("PUPA_API_KEY", "k")
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)

    app = FastAPI()
    app.middleware("http")(run_scope_middleware)
    app.middleware("http")(api_key_middleware)

    @app.post("/")
    async def run() -> dict:  # pragma: no cover - must not be reached
        return {"ok": True}

    # No app.state.run_paths assigned.
    resp = TestClient(app).post("/", json={}, headers={"Authorization": "Bearer k"})
    assert resp.status_code == 503, (
        f"run endpoint answered {resp.status_code} with no scope data — the "
        "fail-closed branch did not fire"
    )
