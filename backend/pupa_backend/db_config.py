"""Database configuration for the LangGraph checkpointer + store.

One env var drives persistence::

    DATABASE_URL=postgresql://user:pass@host:5432/pupa

The URL *scheme* selects the backend — there is no separate discriminator
key. Railway and most managed Postgres providers inject ``DATABASE_URL``
automatically, so a cloud deploy needs no other persistence config; on
Railway, reference the Postgres service (``${{Postgres.DATABASE_URL}}``).

Both the checkpointer and the store bind to that one URL. If it is unset,
each falls back to its own persistent SQLite file under ``~/.pupa-backend/``
(``checkpoints.db`` / ``store.db``) — never in-memory, so local-dev chat
history survives a backend restart. Cloud deploys set
``PUPA_REQUIRE_DB_SCHEME=postgresql`` (via ``persistence.require_db_scheme``
in ``deploy/cloud-config.yml``) to forbid that fallback and fail loudly when
``DATABASE_URL`` is missing.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

SCHEME_SQLITE = "sqlite"
SCHEME_POSTGRES = "postgresql"
SUPPORTED_SCHEMES = {SCHEME_SQLITE, SCHEME_POSTGRES}

# `postgres://` is the legacy alias Railway/Heroku still emit; psycopg only
# accepts `postgresql://`, so it is normalised on the way in.
_SCHEME_ALIASES = {"postgres": SCHEME_POSTGRES}

DATABASE_URL_ENV = "DATABASE_URL"

# Optional hard requirement on the resolved scheme. When set, startup fails
# loudly if no DB is configured at all or if a different backend resolves —
# guarding multi-tenant deploys against silently falling back to SQLite,
# whose data dies with the container.
REQUIRE_DB_SCHEME_ENV = "PUPA_REQUIRE_DB_SCHEME"

CHECKPOINTER_ROLE = "checkpointer"
STORE_ROLE = "store"

# Local-dev SQLite fallback. Distinct files for checkpointer and store keep
# langgraph's two schemas isolated. They live next to config.yml so users can
# find them (and `rm` them to reset local history).
_DEFAULT_SQLITE_DIR = "~/.pupa-backend"
_DEFAULT_SQLITE_FILES = {
    CHECKPOINTER_ROLE: "checkpoints.db",
    STORE_ROLE: "store.db",
}

# Tolerated on input, longest first: `sqlite:///rel/path.db` is the SQLAlchemy
# spelling; `sqlite:////abs/path.db` yields a leading slash after stripping.
_SQLITE_PREFIXES = ("sqlite:///", "sqlite://", "sqlite:")


def scheme_of(url: str) -> str:
    """Return the normalised backend scheme for ``url``."""
    raw = (urlparse(url).scheme or "").lower()
    scheme = _SCHEME_ALIASES.get(raw, raw)
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported database URL scheme {raw!r} — expected one of "
            f"{sorted(SUPPORTED_SCHEMES)} (e.g. postgresql://user:pass@host:5432/db)."
        )
    return scheme


def normalise_url(url: str) -> str:
    """Rewrite a legacy ``postgres://`` URL to the ``postgresql://`` psycopg wants."""
    parsed = urlparse(url)
    raw = (parsed.scheme or "").lower()
    if raw in _SCHEME_ALIASES:
        return url.replace(f"{parsed.scheme}://", f"{_SCHEME_ALIASES[raw]}://", 1)
    return url


def sqlite_path(url: str) -> Path:
    """Expand a ``sqlite://`` URL to a filesystem path, creating its parent dir.

    ``AsyncSqliteSaver.from_conn_string`` takes a path, not a URL.
    """
    raw = url
    for prefix in _SQLITE_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if not raw:
        raise ValueError(f"sqlite URL has no path: {url!r}")
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_postgres_url(url: str) -> None:
    """Fail early on a Postgres URL missing the parts psycopg needs."""
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"{DATABASE_URL_ENV} is missing a host")
    if not (parsed.path or "").lstrip("/"):
        raise ValueError(f"{DATABASE_URL_ENV} is missing a database name")


def load_url(role: str) -> str:
    """Resolve the connection URL for ``role`` (checkpointer or store).

    ``DATABASE_URL`` when set — both roles share it — otherwise the per-role
    persistent SQLite fallback.
    """
    if role not in _DEFAULT_SQLITE_FILES:
        raise ValueError(
            f"Unknown DB role {role!r}; expected one of {sorted(_DEFAULT_SQLITE_FILES)}"
        )

    raw = os.getenv(DATABASE_URL_ENV, "").strip()
    if raw:
        url = normalise_url(raw)
        scheme = scheme_of(url)
        if scheme == SCHEME_POSTGRES:
            validate_postgres_url(url)
        _enforce_required_scheme(role, resolved=scheme)
        return url

    # No explicit config. If the deployment pins a required scheme, fail here
    # — the SQLite fallback below would silently mask a missing DATABASE_URL.
    _enforce_required_scheme(role, resolved=None)
    return f"sqlite:///{_DEFAULT_SQLITE_DIR}/{_DEFAULT_SQLITE_FILES[role]}"


def _enforce_required_scheme(role: str, resolved: str | None) -> None:
    """Validate the resolved scheme against ``PUPA_REQUIRE_DB_SCHEME``."""
    required = os.getenv(REQUIRE_DB_SCHEME_ENV, "").strip().lower()
    if not required:
        return
    required = _SCHEME_ALIASES.get(required, required)
    if required not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"{REQUIRE_DB_SCHEME_ENV}={required!r} is not a supported scheme "
            f"(expected one of {sorted(SUPPORTED_SCHEMES)})."
        )
    if resolved is None:
        raise ValueError(
            f"{REQUIRE_DB_SCHEME_ENV}={required!r} but no database is configured "
            f"for the {role}. Set {DATABASE_URL_ENV} (Railway: reference the "
            "Postgres service as ${{Postgres.DATABASE_URL}})."
        )
    if resolved != required:
        raise ValueError(
            f"{REQUIRE_DB_SCHEME_ENV}={required!r} but {DATABASE_URL_ENV} resolved "
            f"to {resolved!r} for the {role}. Multi-tenant deploys must use the "
            "required backend; SQLite data dies with the container."
        )
