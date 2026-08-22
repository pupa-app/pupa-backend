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


def test_limiter_is_the_outermost_guard(dispatch_stack: list[Any]) -> None:
    """First thing a flood hits, after the header pass. It protects a pre-auth
    route, so it has to run before auth and regardless of the outcome."""
    guards = [m for m in dispatch_stack if m is not security_headers_middleware]
    assert guards[0] is rate_limit_middleware


def test_run_scope_guard_is_mounted_inside_auth(dispatch_stack: list[Any]) -> None:
    assert run_scope_middleware in dispatch_stack, "run-scope guard is not mounted"
    assert api_key_middleware in dispatch_stack
    assert dispatch_stack.index(api_key_middleware) < dispatch_stack.index(run_scope_middleware)


def test_https_guard_is_mounted_outside_auth(dispatch_stack: list[Any]) -> None:
    """Outside auth so a plaintext request is refused before its bearer token
    is read at all, and inside the limiter so a flood still gets throttled
    first."""
    assert require_https_middleware in dispatch_stack, "https guard is not mounted"
    assert dispatch_stack.index(rate_limit_middleware) < dispatch_stack.index(require_https_middleware)
    assert dispatch_stack.index(require_https_middleware) < dispatch_stack.index(api_key_middleware)


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


def test_the_guard_fails_closed_without_run_paths() -> None:
    """If the attribute is missing the guard must refuse, not wave the request
    through — that branch is the difference between a bug and an open door."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pupa_backend.auth import api_key_middleware

    app = FastAPI()
    app.middleware("http")(run_scope_middleware)
    app.middleware("http")(api_key_middleware)

    @app.post("/")
    async def run() -> dict:  # pragma: no cover - must not be reached
        return {"ok": True}

    # No app.state.run_paths assigned.
    resp = TestClient(app).post("/", json={}, headers={"Authorization": "Bearer x"})
    assert resp.status_code != 200, "run endpoint served with no scope data"
