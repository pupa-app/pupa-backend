"""Shared-secret API-key auth for the Pupa backend.

Activated by setting `PUPA_API_KEY` in the environment. When unset, the
middleware is a no-op so localhost dev keeps working without ceremony. When
set, every request must carry `Authorization: Bearer <key>` (constant-time
compared) except for two always-public probes: `GET /auth/config` so clients
can detect whether auth is required, and any `/health` endpoint.

The header shape is deliberately `Authorization: Bearer` so a future swap from
a static shared key to per-user JWTs only changes the verification step —
clients keep sending the same header.
"""

from .middleware import api_key_middleware
from .routes import router
from .scopes import require_api_key, require_scope

__all__ = ["api_key_middleware", "router", "require_api_key", "require_scope"]
