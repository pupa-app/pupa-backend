"""HTTP middleware that gates every route behind either the shared
`PUPA_API_KEY` or a paired device token from
[`devices.py`](devices.py).

**Auth is required by default.** The only way to disable it is the explicit
`PUPA_AUTH_DISABLED=1` opt-out, used for same-laptop dev flows
(`make backend` + `make mac-demo` on one Mac) where the backend isn't
reachable from anywhere else. Production / LAN / tunnel deployments leave
the env var unset and pair their devices.

When auth is required, every non-public request must carry
`Authorization: Bearer <token>` and the token must match one of:
1. A live (non-revoked) paired device token in the `DeviceStore`.
2. The exact `PUPA_API_KEY` value (constant-time compared) — the
   server-side bootstrap credential used by `make pair` to authenticate
   against `/auth/pair/begin` for the very first device. Never given to
   end-user clients; the operator can unset it once at least one device
   is paired.

When matched, the resolved auth identity is attached to `request.state.auth`
as `("device", PairedDevice)` or `("api_key", None)`. Future per-scope
enforcement (Phase 5) reads this.

Registered globally in `app.py` so it covers the opaque AG-UI endpoint at `/`
as well as the `/db`, `/backend-tools`, and `/auth/devices` routers — the
AG-UI helper doesn't expose a dependency-injection seam. A middleware also
runs ahead of Pydantic body parsing, which keeps it forward-compatible with
future request rewrites.
"""

import hmac
import os
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .devices import get_store, truthy


def _is_public(path: str) -> bool:
    if path == "/auth/config":
        return True
    # `/auth/pair` is the bootstrap-code-exchange endpoint — the code IS the
    # credential, so we can't gate it behind a token (the device doesn't have
    # one yet). `/auth/pair/begin` is gated; only `/auth/pair` itself is open.
    if path == "/auth/pair":
        return True
    # The AGUI helper registers `GET {path}/health` — with `path="/"` that
    # serialises to `//health` until Starlette normalises it, so match both.
    if path.endswith("/health"):
        return True
    return False


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": "Invalid or missing API key."},
        headers={"WWW-Authenticate": 'Bearer realm="pupa"'},
    )


async def api_key_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if truthy(os.getenv("PUPA_AUTH_DISABLED")):
        # Explicit opt-out for the same-Mac dev flow. Skip every check.
        return await call_next(request)

    api_key = os.getenv("PUPA_API_KEY")

    if _is_public(request.url.path):
        return await call_next(request)

    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return _unauthorized()

    # Try the paired-device path first — it's the more privileged credential
    # (per-device, revocable) and the new common case.
    device = await get_store().resolve(token)
    if device is not None:
        request.state.auth = ("device", device)
        return await call_next(request)

    # Fallback: legacy shared key. Constant-time compare to avoid timing oracles.
    if api_key and hmac.compare_digest(token, api_key):
        request.state.auth = ("api_key", None)
        return await call_next(request)

    return _unauthorized()
