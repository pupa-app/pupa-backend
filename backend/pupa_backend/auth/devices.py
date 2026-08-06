"""Paired-device token store.

Foundation for the pair-once auth flow (Phase 2). Phase 3 adds the
`/auth/pair` HTTP route + `make pair` CLI; here we just provide the
storage primitives and let the rest of the codebase resolve / list / revoke
device tokens.

Two backends are provided and selected automatically by `get_store()`:

- `PostgresDeviceStore` — used when `DATABASE_URL` is a `postgresql://` URL.
  Creates the `pupa_devices` table if absent. Survives restarts and works
  across replicas.

- `DeviceStore` (JSON file) — fallback for local / SQLite setups. Default path
  `~/.pupa-backend/pupa-auth.json` (stable, CWD-independent — the same home as
  config.yml / TLS / logs); override via `PUPA_AUTH_DB_PATH`.

Tokens are stored as SHA-256 hashes of the raw bearer value. The plaintext
only lives on the device (iOS Keychain) and in the one-shot pairing response
the operator approves. We never log it.

Default scopes when issuing — `agent`, `db`, `tools`, `memory`, `screenshare`
— granted at pair time. Per-scope toggles in the operator UI arrive with
Phase 5 polish; scope enforcement on most routes is also a Phase 5 item.
Phase 2 only enforces the `screenshare` scope (most security-sensitive surface).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Stable home alongside config.yml / TLS certs / logs. Must NOT be
# CWD-relative: the `pupa-backend` console script runs from wherever it's
# invoked, so a relative path would silently point at a different (empty)
# store per launch dir and "unpair" every device. Override via
# PUPA_AUTH_DB_PATH.
DEFAULT_PATH = Path.home() / ".pupa-backend" / "pupa-auth.json"
DEFAULT_SCOPES: tuple[str, ...] = (
    "agent",
    "db",
    "tools",
    "memory",
    "screenshare",
)


@dataclass(frozen=True)
class PairedDevice:
    """Public view of a paired device — never contains the token itself."""

    id: str
    label: str
    scopes: tuple[str, ...]
    created_at: str
    revoked_at: str | None
    # ISO timestamp at which the device's token stops being valid. None for
    # legacy / LAN devices that were issued before TTLs were a thing.
    expires_at: str | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        moment = now or datetime.now(timezone.utc)
        return moment >= datetime.fromisoformat(self.expires_at)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _device_from_entry(entry: dict) -> PairedDevice:
    """Hydrate a JSON-stored entry into a `PairedDevice`. `expires_at` is
    optional so legacy on-disk records (no field) still load.
    """
    return PairedDevice(
        id=entry["id"],
        label=entry["label"],
        scopes=tuple(entry["scopes"]),
        created_at=entry["created_at"],
        revoked_at=entry.get("revoked_at"),
        expires_at=entry.get("expires_at"),
    )


class DeviceStore:
    """Async-safe JSON-backed device registry. One instance per backend
    process; tests reset via `reset_for_testing`.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_PATH
        self._lock = asyncio.Lock()
        # In-memory snapshot. Keyed by token_hash → entry dict so the hot
        # lookup path (every authenticated request) is O(1).
        self._entries: dict[str, dict] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            self._entries = {}
            self._loaded = True
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Corrupt file — start fresh rather than fail hard. The operator
            # is responsible for not nuking their own state; we just don't
            # want a single bad byte to take down the backend.
            self._entries = {}
            self._loaded = True
            return
        self._entries = data.get("devices", {})
        self._loaded = True

    async def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"devices": self._entries}, indent=2, sort_keys=True)
        # Tighten file permissions — the JSON has hashed tokens (still useful
        # for an attacker as a target list). Write to tmp then rename for
        # atomicity.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(payload)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self._path)

    async def issue(
        self,
        label: str,
        scopes: Iterable[str] = DEFAULT_SCOPES,
        ttl: timedelta | None = None,
    ) -> tuple[str, str]:
        """Mint a new device token. Returns `(device_id, plaintext_token)` —
        the only time the plaintext exists outside the device's Keychain.

        When `ttl` is set, the token rejects auth after that duration. None
        (default) keeps the legacy never-expires behaviour.
        """
        token = secrets.token_urlsafe(32)
        token_hash = _hash_token(token)
        device_id = str(uuid.uuid4())
        expires_at = (_now_utc() + ttl).isoformat() if ttl is not None else None
        entry = {
            "id": device_id,
            "label": label,
            "scopes": list(scopes),
            "created_at": _now_iso(),
            "revoked_at": None,
            "expires_at": expires_at,
        }
        async with self._lock:
            await self._ensure_loaded()
            self._entries[token_hash] = entry
            await self._persist()
        return device_id, token

    async def resolve(self, token: str) -> PairedDevice | None:
        """Verify a bearer token against the store. Returns the device on
        success, `None` for unknown / revoked / expired tokens.
        """
        token_hash = _hash_token(token)
        async with self._lock:
            await self._ensure_loaded()
            entry = self._entries.get(token_hash)
        if entry is None or entry.get("revoked_at"):
            return None
        device = _device_from_entry(entry)
        if device.is_expired():
            return None
        return device

    async def list_active(self) -> list[PairedDevice]:
        async with self._lock:
            await self._ensure_loaded()
            snapshot = list(self._entries.values())
        return [
            _device_from_entry(e)
            for e in snapshot
            if not e.get("revoked_at")
        ]

    async def revoke(self, device_id: str) -> bool:
        """Mark a device as revoked. Returns False if no active match."""
        async with self._lock:
            await self._ensure_loaded()
            for entry in self._entries.values():
                if entry["id"] == device_id and not entry.get("revoked_at"):
                    entry["revoked_at"] = _now_iso()
                    await self._persist()
                    return True
        return False


class PostgresDeviceStore:
    """Async Postgres-backed device registry.

    Uses psycopg3 (already a dep via langgraph-checkpoint-postgres). On first
    use it creates `pupa_devices` if absent, then opens a small
    connection pool. All subsequent calls reuse the pool — no lock needed after
    init because pool.connection() is thread/task safe.
    """

    _TABLE = "pupa_devices"
    _DDL = f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            token_hash TEXT PRIMARY KEY,
            id         TEXT NOT NULL,
            label      TEXT NOT NULL,
            scopes     TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            expires_at TEXT
        )
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None
        self._init_lock = asyncio.Lock()

    async def _get_pool(self):
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._pool is not None:
                return self._pool
            from psycopg_pool import AsyncConnectionPool  # type: ignore[import]
            pool = AsyncConnectionPool(self._dsn, open=False, min_size=1, max_size=4)
            await pool.open()
            async with pool.connection() as conn:
                await conn.execute(self._DDL)
            logger.info("PostgresDeviceStore ready (table: %s)", self._TABLE)
            self._pool = pool
        return self._pool

    async def issue(
        self,
        label: str,
        scopes: Iterable[str] = DEFAULT_SCOPES,
        ttl: timedelta | None = None,
    ) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        token_hash = _hash_token(token)
        device_id = str(uuid.uuid4())
        expires_at = (_now_utc() + ttl).isoformat() if ttl is not None else None
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO {self._TABLE} "
                "(token_hash, id, label, scopes, created_at, revoked_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, NULL, %s)",
                (token_hash, device_id, label, json.dumps(list(scopes)), _now_iso(), expires_at),
            )
        return device_id, token

    async def resolve(self, token: str) -> PairedDevice | None:
        token_hash = _hash_token(token)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT id, label, scopes, created_at, revoked_at, expires_at "
                f"FROM {self._TABLE} WHERE token_hash = %s",
                (token_hash,),
            )
            row = await cur.fetchone()
        if row is None or row[4] is not None:  # revoked_at set
            return None
        device = PairedDevice(
            id=row[0],
            label=row[1],
            scopes=tuple(json.loads(row[2])),
            created_at=row[3],
            revoked_at=row[4],
            expires_at=row[5],
        )
        if device.is_expired():
            return None
        return device

    async def list_active(self) -> list[PairedDevice]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT id, label, scopes, created_at, revoked_at, expires_at "
                f"FROM {self._TABLE} WHERE revoked_at IS NULL",
            )
            rows = await cur.fetchall()
        return [
            PairedDevice(
                id=r[0],
                label=r[1],
                scopes=tuple(json.loads(r[2])),
                created_at=r[3],
                revoked_at=r[4],
                expires_at=r[5],
            )
            for r in rows
        ]

    async def revoke(self, device_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"UPDATE {self._TABLE} SET revoked_at = %s "
                "WHERE id = %s AND revoked_at IS NULL",
                (_now_iso(), device_id),
            )
        return cur.rowcount > 0


def _postgres_dsn() -> str | None:
    """Return a Postgres DSN if one is configured for the checkpointer, else None."""
    try:
        from pupa_backend.db_config import (  # type: ignore[import]
            CHECKPOINTER_ROLE,
            SCHEME_POSTGRES,
            load_url,
            scheme_of,
        )
        url = load_url(CHECKPOINTER_ROLE)
        if scheme_of(url) == SCHEME_POSTGRES:
            return url
    except Exception:
        pass
    return None


# Module-level singleton. Tests swap it out via `reset_for_testing`.
_store: DeviceStore | PostgresDeviceStore | None = None


def get_store() -> DeviceStore | PostgresDeviceStore:
    global _store
    if _store is None:
        dsn = _postgres_dsn()
        if dsn:
            logger.info("DeviceStore: using Postgres backend")
            _store = PostgresDeviceStore(dsn)
        else:
            path = os.getenv("PUPA_AUTH_DB_PATH")
            _store = DeviceStore(Path(path)) if path else DeviceStore()
    return _store


def reset_for_testing(path: Path | None = None) -> DeviceStore:
    """Replace the module-level singleton. Tests call this in setup so each
    case gets a fresh JSON file at a tmp path.
    """
    global _store
    _store = DeviceStore(path)
    return _store


def truthy(value: str | None) -> bool:
    """Shared truthy-set so every env-var-driven switch in the auth stack
    (`PUPA_AUTH_DISABLED`, `PUPA_SCREENSHARE`, …) parses the
    same way.
    """
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no")
