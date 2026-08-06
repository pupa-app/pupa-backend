"""The installed backend version.

A packaging fact, not a tracing or auth one: `GET /auth/config` reports it so
a client can pin a compatible backend, and Langfuse stamps traces with it.
"""

import importlib.metadata


def backend_version() -> str:
    """Return the installed `pupa-backend` version, or ``"unknown"``.

    ``"unknown"`` covers running from a source checkout that was never
    installed — the server still boots, it just can't name its own build.
    """
    try:
        return importlib.metadata.version("pupa-backend")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
