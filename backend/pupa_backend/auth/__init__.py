"""Auth, authorization, and transport guards for the Pupa backend.

**Auth is required by default.** Every request must carry
`Authorization: Bearer <token>`, matching either a live paired-device token or
the operator `PUPA_API_KEY`, except for three public probes: `GET /auth/config`
(so clients can detect whether auth is required), `POST /auth/pair` (the
bootstrap code *is* the credential at that point), and any `/health` endpoint.
`PUPA_AUTH_DISABLED=1` turns the whole thing off — same-machine dev only.

The pieces, outermost first as they run:

- `security_headers_middleware` — nosniff / frame-deny / referrer, HSTS on TLS.
- `rate_limit_middleware` — per-client throttle on the pairing routes.
- `require_https_middleware` — refuses plaintext when `PUPA_REQUIRE_HTTPS` is set.
- `api_key_middleware` — resolves the caller into `request.state.auth`.
- `run_scope_middleware` — `agent` scope on the harness run endpoints.
- `require_scope` / `require_api_key` — per-route dependencies.
"""

from .middleware import api_key_middleware, run_scope_middleware
from .headers import security_headers_middleware
from .ratelimit import rate_limit_middleware
from .transport import is_secure_request, require_https_middleware
from .routes import router
from .scopes import require_api_key, require_scope

__all__ = [
    "api_key_middleware",
    "is_secure_request",
    "rate_limit_middleware",
    "require_https_middleware",
    "run_scope_middleware",
    "security_headers_middleware",
    "router",
    "require_api_key",
    "require_scope",
]
