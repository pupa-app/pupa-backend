"""Tests for `db_config` — the single `DATABASE_URL` convention.

The URL scheme is the backend discriminator; there is no `db_type` key and
no per-role override. Unset `DATABASE_URL` means the local SQLite fallback,
which `PUPA_REQUIRE_DB_SCHEME` can forbid.
"""

import pytest

from pupa_backend.db_config import (
    CHECKPOINTER_ROLE,
    SCHEME_POSTGRES,
    SCHEME_SQLITE,
    STORE_ROLE,
    load_url,
    normalise_url,
    scheme_of,
    sqlite_path,
)


def _clear_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DATABASE_URL", "PUPA_REQUIRE_DB_SCHEME"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Scheme parsing — the discriminator
# ---------------------------------------------------------------------------

def test_scheme_of_postgresql() -> None:
    assert scheme_of("postgresql://alice:secret@db.example.com:5433/pupa") == SCHEME_POSTGRES


def test_scheme_of_normalises_legacy_postgres_alias() -> None:
    """Railway/Heroku still emit `postgres://`; psycopg only takes `postgresql://`."""
    assert scheme_of("postgres://u:p@h:5432/d") == SCHEME_POSTGRES
    assert normalise_url("postgres://u:p@h:5432/d") == "postgresql://u:p@h:5432/d"


def test_normalise_url_leaves_postgresql_untouched() -> None:
    url = "postgresql://u:p@h:5432/d"
    assert normalise_url(url) == url


def test_scheme_of_sqlite() -> None:
    assert scheme_of("sqlite:///./checkpoints.db") == SCHEME_SQLITE


def test_scheme_of_rejects_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported database URL scheme"):
        scheme_of("mysql://u:p@h:3306/d")


# ---------------------------------------------------------------------------
# SQLite URL → filesystem path
# ---------------------------------------------------------------------------

def test_sqlite_path_expands_user(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = sqlite_path("sqlite:///~/.pupa-backend/checkpoints.db")
    assert path == tmp_path / ".pupa-backend" / "checkpoints.db"
    # Parent is created so the saver can open the file.
    assert path.parent.is_dir()


def test_sqlite_path_absolute_four_slash_form(tmp_path) -> None:
    target = tmp_path / "nested" / "abs.db"
    assert sqlite_path(f"sqlite:///{target}") == target


def test_sqlite_path_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="no path"):
        sqlite_path("sqlite:///")


# ---------------------------------------------------------------------------
# load_url — DATABASE_URL drives both roles
# ---------------------------------------------------------------------------

def test_database_url_drives_both_checkpointer_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h:5432/d")
    chk = load_url(CHECKPOINTER_ROLE)
    store = load_url(STORE_ROLE)
    # Both bind to the one URL, normalised for psycopg.
    assert chk == store == "postgresql://u:p@h:5432/d"


def test_database_url_requires_a_database_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/")
    with pytest.raises(ValueError, match="missing a database name"):
        load_url(CHECKPOINTER_ROLE)


def test_database_url_requires_a_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql:///justadb")
    with pytest.raises(ValueError, match="missing a host"):
        load_url(CHECKPOINTER_ROLE)


def test_unsupported_database_url_scheme_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "mysql://u:p@h:3306/d")
    with pytest.raises(ValueError, match="Unsupported database URL scheme"):
        load_url(CHECKPOINTER_ROLE)


def test_no_config_falls_back_to_distinct_sqlite_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-dev default: persistent SQLite under ~/.pupa-backend/ rather than
    in-memory, so chat history survives a backend restart. Distinct files keep
    langgraph's checkpointer and store schemas from colliding.
    """
    _clear_db_env(monkeypatch)
    chk = load_url(CHECKPOINTER_ROLE)
    store = load_url(STORE_ROLE)
    assert scheme_of(chk) == scheme_of(store) == SCHEME_SQLITE
    assert chk.endswith("checkpoints.db")
    assert store.endswith("store.db")
    assert chk != store


def test_unknown_role_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    with pytest.raises(ValueError, match="Unknown DB role"):
        load_url("cache")


# ---------------------------------------------------------------------------
# PUPA_REQUIRE_DB_SCHEME — hard requirement for multi-tenant deploys
# ---------------------------------------------------------------------------

def test_require_scheme_fails_when_no_db_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("PUPA_REQUIRE_DB_SCHEME", "postgresql")
    with pytest.raises(ValueError, match="no database is configured"):
        load_url(CHECKPOINTER_ROLE)


def test_require_scheme_fails_when_resolved_scheme_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("PUPA_REQUIRE_DB_SCHEME", "postgresql")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")
    with pytest.raises(ValueError, match="resolved to 'sqlite'"):
        load_url(CHECKPOINTER_ROLE)


def test_require_scheme_passes_when_postgres_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("PUPA_REQUIRE_DB_SCHEME", "postgresql")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h:5432/d")
    assert load_url(CHECKPOINTER_ROLE) == "postgresql://u:p@h:5432/d"


def test_require_scheme_accepts_legacy_postgres_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`require_db_scheme: postgres` in an older config.yml still works."""
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("PUPA_REQUIRE_DB_SCHEME", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/d")
    assert load_url(CHECKPOINTER_ROLE) == "postgresql://u:p@h:5432/d"


def test_require_scheme_rejects_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("PUPA_REQUIRE_DB_SCHEME", "mongodb")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h:5432/d")
    with pytest.raises(ValueError, match="not a supported scheme"):
        load_url(CHECKPOINTER_ROLE)


def test_require_scheme_unset_allows_sqlite_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-dev with no DB config gets the SQLite fallback, not an error."""
    _clear_db_env(monkeypatch)
    assert scheme_of(load_url(CHECKPOINTER_ROLE)) == SCHEME_SQLITE
