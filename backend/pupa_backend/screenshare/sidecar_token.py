"""Per-process bearer token for the local screen-share sidecar.

Generated fresh on every backend startup. Written to a temp file so
`pupa-backend screenshare` can read it without manual copy-paste.  File
permissions are 0o600 (owner-read only).

Using a file rather than a loopback-IP check keeps auth sound even
when the backend runs behind a local reverse proxy (ngrok, cloudflared)
whose connections also arrive from 127.0.0.1.
"""

import hmac
import os
import secrets

TOKEN_PATH = "/tmp/pupa-sidecar.token"

_token: str | None = None


def generate() -> str:
    """Generate a fresh secret, persist it, and return it."""
    global _token
    _token = secrets.token_hex(32)
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(_token)
    os.chmod(TOKEN_PATH, 0o600)
    return _token


def validate(candidate: str) -> bool:
    """Return True iff candidate matches the in-process sidecar secret."""
    return _token is not None and hmac.compare_digest(candidate, _token)


def revoke() -> None:
    """Zero out the in-process token and delete the temp file."""
    global _token
    _token = None
    try:
        os.unlink(TOKEN_PATH)
    except FileNotFoundError:
        pass
