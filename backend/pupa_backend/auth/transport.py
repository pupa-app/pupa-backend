"""Transport-security enforcement — `PUPA_REQUIRE_HTTPS`.

TLS is optional by design: `pupa-backend` is self-hosted first, and forcing
HTTPS would break offline and LAN installs that have no cert and no name to
put one on. So the switch is the operator's, mirroring
`PUPA_REQUIRE_DB_SCHEME`: unset by default, set on anything reachable from
the internet.

What it protects: `/auth/pair` hands the device token to the client in
plaintext exactly once, and every authenticated request after that carries a
bearer token. On a plaintext hop, both are readable by anyone on the path.
"""

import os
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .devices import truthy
from .paths import is_health_probe
from .proxy import forwarded_values, trust_forwarded_headers


def require_https_enabled() -> bool:
    return truthy(os.getenv("PUPA_REQUIRE_HTTPS"))


def is_secure_request(request: Request) -> bool:
    """Whether the caller's hop was actually TLS.

    Two signals, because the answer differs by deployment:
    - `url.scheme` — the backend terminating its own TLS (`PUPA_TLS_CERT`).
    - `X-Forwarded-Proto` — every tunnel mode (Tailscale serve, Cloudflare
      tunnel, Railway) terminates TLS in front of a loopback-bound listener,
      so this process only ever sees `http` and the header is the sole
      evidence of the real hop.

    Rightmost entry wins: a caller can forge the header, but the trusted proxy
    appends what it actually saw — and the header is only consulted at all
    when `trust_forwarded_headers()` says a proxy is in front. Without that
    check a client could assert its own plaintext hop was TLS and walk
    straight through `PUPA_REQUIRE_HTTPS`.

    Note what is *not* here: any check on the peer address. Because those
    proxies all connect over loopback, "it came from 127.0.0.1" is true of
    every remote caller, so a loopback carve-out would exempt the internet.
    """
    if request.url.scheme in ("https", "wss"):
        return True
    if trust_forwarded_headers():
        hops = forwarded_values(request.headers, "x-forwarded-proto")
        if hops:
            return hops[-1].lower() in ("https", "wss")
    return False


async def require_https_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Refuse plaintext when `PUPA_REQUIRE_HTTPS` is set. No-op otherwise."""
    if not require_https_enabled():
        return await call_next(request)
    # Platform health probes hit the listener directly, behind the TLS
    # terminator; blocking them would fail the deploys this flag exists for.
    if is_health_probe(request.url.path) or is_secure_request(request):
        return await call_next(request)
    # Read by `rate_limit_middleware`, which wraps this one: a plaintext hop
    # is a misconfiguration, not a guess at a credential, so it must not spend
    # a legitimate device's pairing budget.
    request.state.transport_refused = True
    return JSONResponse(
        status_code=403,
        content={
            "detail": (
                "HTTPS required. This backend runs with PUPA_REQUIRE_HTTPS set "
                "and refuses plaintext requests — terminate TLS in front of it."
            )
        },
    )
