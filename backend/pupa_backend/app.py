"""Pupa FastAPI app — exposes the A2UI agent over AG-UI.

The lifespan opens the LangGraph checkpointer + store from `DATABASE_URL` (or
local SQLite when unset) and backs both the agent graph and the `/db` REST
routes with them.
"""

import logging
import os
import re
import shutil
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load ~/.pupa-backend/config.yml (or .env legacy) — shell env takes precedence.
from pupa_backend.pupa_config import load_pupa_config
load_pupa_config()

# Load project-local .env (override=True so local dev keys beat global env).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from pupa_backend.version import backend_version

from pupa_backend.credentials import stash_forbidden_credentials
from pupa_backend.harnesses import (
    HarnessDeps,
    build_registry,
    claude_harness_enabled,
    deepagents_harness_enabled,
)
from pupa_backend.mcp_servers import mcp_servers_lifecycle
from pupa_backend.auth import (
    api_key_middleware,
    rate_limit_middleware,
    require_https_middleware,
    run_scope_middleware,
    security_headers_middleware,
)
from pupa_backend.auth import router as auth_router
from pupa_backend.sse_keepalive import SSEKeepAliveMiddleware
from pupa_backend.sse_replay import SSEReplayMiddleware
from pupa_backend.auth.devices import get_store as get_device_store, truthy
from pupa_backend.harnesses.routes import router as harnesses_router
from pupa_backend.screenshare.config import is_enabled as screenshare_enabled


# Piggyback on uvicorn's pre-configured handler so messages actually show up
# in `make backend` output. Matches the pattern in agent.py — every other
# pupa log line goes through this same logger.
logger = logging.getLogger("uvicorn.error")


# When the Claude Code harness is enabled, move billing-diverting credential vars
# out of os.environ into the in-process stash BEFORE anything spawns the `claude`
# subprocess, so it can't inherit them (see credentials.py). The LangGraph model
# builders read the stash via `get_credential`. Runs after config + .env load.
if claude_harness_enabled():
    stash_forbidden_credentials()


_TUNNEL_URL_FILE = Path.home() / ".pupa-backend" / "tunnel_url"

# Bind addresses only this machine can reach. Anything else — `0.0.0.0`, or an
# explicit `PUPA_HOST` on a LAN address — is reachable by something that isn't
# the operator, which is what `PUPA_AUTH_DISABLED` must never be paired with.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _start_cloudflared_tunnel() -> "subprocess.Popen[str] | None":
    """Start a cloudflared quick tunnel, write URL to _TUNNEL_URL_FILE, return process."""
    if not shutil.which("cloudflared"):
        logger.warning("connectivity=cloudflared but cloudflared not found — install: brew install cloudflared")
        return None

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8004"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    found = threading.Event()

    def _read(stream: "subprocess.IO[str]") -> None:
        for line in stream:
            m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
            if m and not found.is_set():
                url = m.group(0)
                _TUNNEL_URL_FILE.write_text(url)
                logger.info("cloudflare tunnel ready: %s", url)
                found.set()

    threading.Thread(target=_read, args=(proc.stdout,), daemon=True).start()

    if not found.wait(timeout=30):
        proc.terminate()
        logger.warning("cloudflared did not produce a tunnel URL within 30 s")
        return None

    return proc


def _start_named_tunnel(name: str, hostname: str) -> "subprocess.Popen[str] | None":
    """Start a persistent Cloudflare *named* tunnel (`cloudflared tunnel run
    <name>`) as a managed child of the backend, so `pupa-backend run` brings the
    stable public URL up on its own. Unlike the quick tunnel the URL is known
    up-front (the configured hostname), so we record it immediately and just
    drain cloudflared's output to the log (an unread pipe would eventually block
    the connector)."""
    if not shutil.which("cloudflared"):
        logger.warning(
            "connectivity=cloudflared (named tunnel %r) but cloudflared not found "
            "— install: brew install cloudflared",
            name,
        )
        return None

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "run", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = f"https://{hostname}"
    _TUNNEL_URL_FILE.write_text(url)

    def _drain(stream: "subprocess.IO[str]") -> None:
        for line in stream:
            logger.info("cloudflared: %s", line.rstrip())

    threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
    logger.info("cloudflare named tunnel starting: %s (tunnel %s)", url, name)
    return proc


async def _log_auth_state() -> None:
    """One-line summary at startup so a fresh `make backend` with no env
    vars doesn't silently look broken when every gated route 401s.
    """
    if truthy(os.getenv("PUPA_AUTH_DISABLED")):
        logger.warning(
            "auth DISABLED via PUPA_AUTH_DISABLED=1 — every route is open. "
            "Never set this on a backend reachable from anywhere off-laptop."
        )
        return

    api_key_set = bool(os.getenv("PUPA_API_KEY"))
    paired_count = len(await get_device_store().list_active())

    if api_key_set and paired_count == 0:
        logger.info(
            "auth required — bootstrap key set, no paired devices yet. "
            "Run `make pair LABEL=...` to mint a code, paste it into iOS Settings → Backend → Edit."
        )
    elif api_key_set and paired_count > 0:
        logger.info(
            "auth required — bootstrap key set, %d paired device(s). "
            "Once you trust the existing devices you can unset PUPA_API_KEY.",
            paired_count,
        )
    elif not api_key_set and paired_count > 0:
        logger.info(
            "auth required — %d paired device(s); bootstrap key not set. "
            "Set PUPA_API_KEY temporarily if you need to pair a new device.",
            paired_count,
        )
    else:
        logger.warning(
            "auth required but nothing is configured — every gated route will return 401. "
            "Either set PUPA_API_KEY=<value> and restart to bootstrap pairing, "
            "or set PUPA_AUTH_DISABLED=1 for same-laptop dev only."
        )


@asynccontextmanager
async def _persistence_lifespan() -> AsyncGenerator[tuple[Any, Any], None]:
    """Open the checkpointer + store — but only for the deepagents harness.

    Both the savers and the `/db` router read and write that loop's LangGraph
    checkpoints, so a Claude-only deploy has nothing for them to serve: that
    loop keeps its sessions in-process and the SDK owns its own history.
    Skipping the open also skips the `PUPA_REQUIRE_DB_SCHEME` fail-fast, which
    would otherwise demand a database no harness in the process would read.
    """
    if not deepagents_harness_enabled():
        yield None, None
        return
    from pupa_backend.db_config import CHECKPOINTER_ROLE, STORE_ROLE, load_url
    from pupa_backend.harnesses.langgraph.db import open_persistence

    async with open_persistence(
        load_url(CHECKPOINTER_ROLE), load_url(STORE_ROLE)
    ) as (checkpointer, store):
        yield checkpointer, store


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("LANGFUSE_RELEASE", f"pupa-backend@{backend_version()}")
    tunnel_proc = None
    if os.getenv("PUPA_CONNECTIVITY") == "cloudflared":
        named_host = os.getenv("PUPA_CLOUDFLARED_HOSTNAME")
        named_tunnel = os.getenv("PUPA_CLOUDFLARED_TUNNEL")
        if named_host and named_tunnel:
            # Named tunnel (custom domain) → bring the stable URL up as a managed
            # child so `pupa-backend run` starts everything together.
            tunnel_proc = _start_named_tunnel(named_tunnel, named_host)
        else:
            tunnel_proc = _start_cloudflared_tunnel()
    async with mcp_servers_lifecycle() as mcp:
        async with _persistence_lifespan() as (checkpointer, store):
            # Mount every enabled harness at POST /harnesses/{id}; the default one
            # is also aliased at POST / (curl smoke + un-migrated clients). The
            # credential scrub above lets the LangGraph and Claude Code harnesses
            # coexist: AWS_*/ANTHROPIC_API_KEY are in the stash, not os.environ, so
            # the `claude` subprocess's billing guard passes while the LangGraph
            # builders still get their creds via `get_credential`.
            registry = build_registry()
            deps = HarnessDeps(checkpointer=checkpointer, store=store, mcp=mcp)
            # Every mounted run path, for `run_scope_middleware`. The routes
            # themselves can't take a scope dependency (the LangGraph one is
            # mounted by a third-party helper), so the guard keys off this set.
            run_paths: set[str] = set()
            for harness in registry.enabled():
                path = f"/harnesses/{harness.id}"
                harness.register(app, path, deps)
                run_paths.add(path)
            default = registry.default()
            if default is not None:
                default.register(app, "/", deps)
                run_paths.add("/")
            app.state.run_paths = run_paths
            app.state.harness_registry = registry
            app.state.checkpointer = checkpointer
            await _log_auth_state()
            if screenshare_enabled():
                from pupa_backend.screenshare.sidecar_token import TOKEN_PATH
                from pupa_backend.screenshare.sidecar_token import generate as _gen_sidecar_token
                _gen_sidecar_token()
                logger.info("screenshare sidecar token → %s", TOKEN_PATH)
            try:
                yield
            finally:
                if screenshare_enabled():
                    from pupa_backend.screenshare.sidecar_token import revoke as _revoke_sidecar_token
                    _revoke_sidecar_token()
                if tunnel_proc is not None:
                    tunnel_proc.terminate()
                    _TUNNEL_URL_FILE.unlink(missing_ok=True)


app = FastAPI(lifespan=lifespan)
# Transport-level resumable SSE: detaches every `POST /` run stream into a
# per-thread sequenced replay log so a dropped socket (app backgrounded /
# killed) can re-attach and catch up instead of losing the turn. Covers both
# agent loops identically — neither loop's code knows about it. ORDER MATTERS:
# added FIRST so it sits innermost (under the keep-alive), which keeps
# heartbeat comments out of the replay log while idle re-attached streams
# still receive them. See sse_replay.py.
app.add_middleware(SSEReplayMiddleware)
# Transport-level SSE keep-alive: emits `: keep-alive` comments on any idle
# `text/event-stream` response so a long silent turn doesn't trip the client's
# per-request idle timeout. Decoupled from the agent loop — covers both `POST /`
# handlers (Claude Code loop + LangGraph) and any future SSE route.
app.add_middleware(SSEKeepAliveMiddleware)
# Inner to `api_key_middleware` (added after it here = added earlier = inner):
# it reads the `request.state.auth` that auth puts there.
app.middleware("http")(run_scope_middleware)
app.middleware("http")(api_key_middleware)
# Outside auth: throttles the pre-auth pairing exchange before any auth work
# happens, and regardless of how that auth turns out.
app.middleware("http")(rate_limit_middleware)
# Outside the limiter: a plaintext hop is a misconfiguration, not a guess at a
# credential, so it must be refused before it can spend a real device's pairing
# budget. No-op unless PUPA_REQUIRE_HTTPS is set.
app.middleware("http")(require_https_middleware)
# Outermost of all: every response gets the headers, including the ones the
# guards above return on their own (403/429 never reach the inner stack).
app.middleware("http")(security_headers_middleware)
app.include_router(auth_router, prefix="/auth")
if deepagents_harness_enabled():
    # Mounted only with its harness — the routes serve that loop's checkpoints.
    # Imported here so a Claude-only process never pulls in langgraph at all.
    from pupa_backend.harnesses.langgraph.db import router as db_router

    app.include_router(db_router, prefix="/db")
app.include_router(harnesses_router, prefix="/harnesses")

if screenshare_enabled():
    from pupa_backend.screenshare import router as screenshare_router

    app.include_router(screenshare_router, prefix="/screenshare")


def main() -> None:
    """Boot the backend under uvicorn. Entrypoint for `pupa-backend run` and
    for `python -m pupa_backend`."""
    import uvicorn

    # Railway (and most PaaS providers) inject the port to bind via $PORT.
    # Local dev keeps the historic 8004 when the var is unset.
    port = int(os.getenv("PORT", "8004"))

    # With `connectivity: tailscale`, let tailscaled forward the tailnet into a
    # loopback-bound listener — a wildcard bind is unreachable on macOS without
    # the Local Network privacy grant. See tailscale_proxy.py.
    from pupa_backend.auth.proxy import trust_forwarded_headers
    from pupa_backend.tailscale_proxy import bind_host, start_serve_proxy

    proxy = start_serve_proxy(port)
    connectivity = os.getenv("PUPA_CONNECTIVITY", "").strip().lower()
    cloudflared = connectivity == "cloudflared"

    # Two facts, and they come apart — `PUPA_CONNECTIVITY` says what was
    # *intended*, not what actually came up:
    #
    # - `fronted`: something local forwards into this listener, so it doesn't
    #   need to be reachable directly. `start_serve_proxy` returns None when the
    #   Tailscale CLI is absent, when `PUPA_TAILSCALE_SERVE=0`, or when `serve`
    #   fails — all documented as falling back to `0.0.0.0`.
    # - `rewrites_forwarded`: that something is an HTTP proxy, so it *writes*
    #   `X-Forwarded-*` over whatever the caller sent. Tailscale's `tcp` mode is
    #   a raw L4 passthrough: the client's request arrives byte-for-byte, so
    #   those headers stay caller-written and must not be believed.
    fronted = proxy is not None or cloudflared
    rewrites_forwarded = cloudflared or (proxy is not None and proxy.terminates_tls)

    if connectivity == "tailscale" and proxy is None:
        logger.warning(
            "connectivity=tailscale but `tailscale serve` is not active — "
            "binding %s and trusting no forwarded headers.",
            bind_host(proxied=False),
        )

    # `auth/proxy.py` infers trust from `PUPA_CONNECTIVITY` for processes that
    # never run this function (a bare `uvicorn pupa_backend.app:app`). Here the
    # answer is known, so record it — an explicit operator setting still wins.
    if not os.getenv("PUPA_TRUSTED_PROXY", "").strip():
        os.environ["PUPA_TRUSTED_PROXY"] = "1" if rewrites_forwarded else "0"

    tls_cert = os.getenv("PUPA_TLS_CERT")
    tls_key = os.getenv("PUPA_TLS_KEY")
    ssl_kwargs: dict = {}
    if proxy is not None and proxy.terminates_tls:
        # tailscaled holds a real, auto-renewing cert for the MagicDNS name and
        # proxies plain HTTP here over loopback — serving our own TLS underneath
        # it would only break that hop.
        logger.info("TLS handled by tailscale serve — backend speaks HTTP on loopback.")
    elif tls_cert and tls_key:
        ssl_kwargs = {"ssl_certfile": tls_cert, "ssl_keyfile": tls_key}
        logger.info("TLS enabled — cert=%s key=%s", tls_cert, tls_key)
        from pupa_backend.tls_check import warn_unusable_cert

        warn_unusable_cert(tls_cert)

    host = bind_host(proxied=fronted)

    if truthy(os.getenv("PUPA_AUTH_DISABLED")) and host not in _LOOPBACK_HOSTS:
        # Refuse, don't warn. `PUPA_AUTH_DISABLED=1` opens every route
        # including the agent loop and the shell tool, and a warning scrolls
        # past in a platform log — the two together are how a dev shortcut ends
        # up serving a stranger. The escape hatch is deliberately a second,
        # differently-named variable so it can't be reached by pasting the
        # first one into a launch script.
        if not truthy(os.getenv("PUPA_ALLOW_INSECURE_BIND")):
            raise SystemExit(
                f"Refusing to start: PUPA_AUTH_DISABLED=1 with the listener "
                f"bound to {host}, which is reachable from off this machine. "
                f"Every route would be open to anyone who can route to it. "
                f"Pair a device instead (see `pupa-backend pair`), bind "
                f"127.0.0.1, or set PUPA_ALLOW_INSECURE_BIND=1 if this network "
                f"really is trusted."
            )
        logger.warning(
            "auth is DISABLED and the listener is bound to %s — every route is "
            "open to anything that can reach this port. Allowed only because "
            "PUPA_ALLOW_INSECURE_BIND is set.",
            host,
        )

    if trust_forwarded_headers() and host not in _LOOPBACK_HOSTS:
        # Not fatal: Railway and friends need the wildcard bind and do sanitise
        # the headers. But anyone who can reach the port directly can now write
        # their own hop, so it has to be a deliberate choice, not a default.
        logger.warning(
            "trusting X-Forwarded-* while bound to %s — anyone who can reach "
            "this port directly can forge their own hop. Front it with the "
            "proxy, or unset transport.trusted_proxy.",
            host,
        )

    try:
        uvicorn.run(
            "pupa_backend.app:app",
            host=host,
            port=port,
            reload=False,
            # uvicorn's own proxy-header handling folds client-supplied
            # `X-Forwarded-Proto`/`-For` into the ASGI scope for any peer in
            # `forwarded_allow_ips` (default `127.0.0.1` — every tunnel mode).
            # That would rewrite `url.scheme` and `client.host` *underneath*
            # `auth/proxy.py`, which is the one place allowed to decide whether
            # those headers are believable. Off: we read the raw headers.
            proxy_headers=False,
            **ssl_kwargs,
        )
    finally:
        if proxy is not None:
            proxy.stop()


if __name__ == "__main__":
    main()
