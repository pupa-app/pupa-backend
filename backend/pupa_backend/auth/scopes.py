"""Per-route authorization helpers.

The auth middleware (:mod:`auth.middleware`) gates *whether* a request is
authenticated. These helpers gate *what an authenticated identity can do*:

- ``require_scope("agent")`` — passes for the ``PUPA_API_KEY`` operator
  bearer; for a paired-device bearer, checks ``device.has_scope("agent")``
  and 403s otherwise.
- ``require_api_key()`` — operator-only routes. Rejects device bearers
  outright (403) even if the device is valid. Used for device management
  (``/auth/devices/*``) and any surface that's cross-user without namespace
  isolation.

Both also honour ``PUPA_AUTH_DISABLED=1`` — the same-laptop dev opt-out
short-circuits the middleware, so these dependencies treat that as
"allow everything" for consistency.

Identity comes from ``request.state.auth`` set by
:func:`auth.middleware.api_key_middleware`:

- ``("api_key", None)`` — operator (bypasses scope checks).
- ``("device", PairedDevice)`` — scoped user.

Anything else means the middleware didn't run or didn't authenticate —
treated as 401 so we never silently allow an unauthenticated caller.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, status

from .devices import truthy


def _auth_disabled() -> bool:
    return truthy(os.getenv("PUPA_AUTH_DISABLED"))


def _identity(request: Request) -> tuple[str, Any] | None:
    return getattr(request.state, "auth", None)


def _forbid(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
    )


def require_scope(scope: str) -> Callable[[Request], None]:
    """FastAPI dependency: caller must be ``api_key`` or a device with ``scope``.

    Routes use it as ``Depends(require_scope("agent"))``. The dependency is
    re-instantiated per call so each route can declare its own scope without
    closure-sharing surprises.
    """

    def _dep(request: Request) -> None:
        if _auth_disabled():
            return
        identity = _identity(request)
        if identity is None:
            raise _unauthorized()
        kind, principal = identity
        if kind == "api_key":
            return
        if kind == "device" and principal is not None and principal.has_scope(scope):
            return
        raise _forbid(f"Missing required scope: {scope!r}")

    return _dep


def require_api_key() -> Callable[[Request], None]:
    """FastAPI dependency: caller must be the ``PUPA_API_KEY`` operator.

    Used for routes that are operator-only (``/auth/devices/*``) or that are
    structurally cross-user without isolation. A valid device token gets a
    clean 403 with a message pointing at the limitation.
    """

    def _dep(request: Request) -> None:
        if _auth_disabled():
            return
        identity = _identity(request)
        if identity is None:
            raise _unauthorized()
        kind, _ = identity
        if kind == "api_key":
            return
        raise _forbid(
            "This route is operator-only (requires PUPA_API_KEY). "
            "Paired-device tokens are not accepted because the surface is "
            "cross-user without isolation."
        )

    return _dep
