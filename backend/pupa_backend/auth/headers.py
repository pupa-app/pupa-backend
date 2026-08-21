"""Response security headers.

Low real-world risk here — the client is a native app, so there's no browser
origin, no cookie, and no frame to clickjack. They're cheap, though, and the
moment anything reaches this API from a browser (a debug console, a future web
client, a proxy that renders errors) they start doing work.

Deliberately **no** `CORSMiddleware`: there is no browser origin to allow
today, and a permissive one would hand any web page the ability to call this
backend with a user's credentials.
"""

import os
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.responses import Response

from .transport import is_secure_request, require_https_enabled

_STATIC_HEADERS = {
    # Don't let a response body be re-interpreted as a type it didn't declare.
    "X-Content-Type-Options": "nosniff",
    # Nothing here is meant to be framed.
    "X-Frame-Options": "DENY",
    # Backend URLs can carry a tunnel hostname; don't leak them onward.
    "Referrer-Policy": "no-referrer",
}

# Two years, the usual preload-eligible value. Only sent on a connection that
# is actually TLS — asserting HSTS over plaintext is meaningless, and pinning a
# LAN host to HTTPS it doesn't serve would lock the operator out of their own
# backend for the lifetime of the max-age.
_HSTS = "max-age=63072000; includeSubDomains"


def _tls_active() -> bool:
    return bool(os.getenv("PUPA_TLS_CERT") and os.getenv("PUPA_TLS_KEY"))


async def security_headers_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response = await call_next(request)
    for header, value in _STATIC_HEADERS.items():
        response.headers.setdefault(header, value)
    if (_tls_active() or require_https_enabled()) and is_secure_request(request):
        response.headers.setdefault("Strict-Transport-Security", _HSTS)
    return response
