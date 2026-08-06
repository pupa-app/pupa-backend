"""Bootstrap-code store for pair-once device auth.

Phase 3a of the pairing work. The
operator runs `make pair` which hits `POST /auth/pair/begin` (admin auth,
behind the existing middleware) and gets back a short-lived 8-character
bootstrap code. The code is shown to the user — printed in the terminal,
optionally rendered as a QR by the iOS app's scanner in Phase 3b. The user's
device hits `POST /auth/pair` with `{code, label}` and gets a permanent
device token in exchange. The code is single-use and expires after 5 minutes
(default).

Storage is *in-memory only* — codes don't survive backend restarts on
purpose. They're meant to live for the few seconds between minting and being
scanned. If the operator restarts the backend mid-pair, they re-run
`make pair`. Persisting would just widen the attack window.

Concurrency: the store is guarded by an `asyncio.Lock` so concurrent mints
and concurrent consumes don't race on the dict.
"""

from __future__ import annotations

import asyncio
import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .devices import DEFAULT_SCOPES

# 8 alphanumeric chars, unambiguous set (no 0/O/1/I/l). 32^8 ≈ 1.1 × 10^12
# codes — easily unique for a single-operator backend and short enough to
# read aloud over the phone if QR scanning isn't available.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LENGTH = 8

DEFAULT_TTL = timedelta(minutes=5)


@dataclass(frozen=True)
class PairingCode:
    """A live bootstrap code waiting to be exchanged for a device token."""

    code: str
    scopes: tuple[str, ...]
    expires_at: datetime
    suggested_label: str | None
    # When set, the device token issued from this code expires after this
    # duration. None → token never expires (legacy behaviour, fine for LAN/
    # same-laptop setups; cloud deployments should set a finite TTL).
    device_token_ttl: timedelta | None = None

    @property
    def expires_at_iso(self) -> str:
        return self.expires_at.isoformat()


def _gen_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


class PairingCodeStore:
    """In-memory single-use bootstrap-code registry. One per backend process."""

    def __init__(self) -> None:
        self._codes: dict[str, PairingCode] = {}
        self._lock = asyncio.Lock()

    async def mint(
        self,
        suggested_label: str | None = None,
        scopes: Iterable[str] = DEFAULT_SCOPES,
        ttl: timedelta = DEFAULT_TTL,
        device_token_ttl: timedelta | None = None,
    ) -> PairingCode:
        """Create a new bootstrap code. The operator displays this to the
        device user (printed by `make pair`, rendered as a QR in Phase 3b).
        """
        async with self._lock:
            # Re-roll on the rare collision. 32^8 makes this practically
            # never happen, but the check costs nothing.
            for _ in range(10):
                code = _gen_code()
                if code not in self._codes:
                    break
            else:
                raise RuntimeError("could not allocate a unique pairing code")
            entry = PairingCode(
                code=code,
                scopes=tuple(scopes),
                expires_at=datetime.now(timezone.utc) + ttl,
                suggested_label=suggested_label,
                device_token_ttl=device_token_ttl,
            )
            self._codes[code] = entry
            self._sweep_expired_locked()
            return entry

    async def consume(self, code: str) -> PairingCode | None:
        """Validate a code, remove it from the store, and return its metadata.
        Subsequent calls with the same code return None — codes are single-use.
        """
        async with self._lock:
            self._sweep_expired_locked()
            entry = self._codes.pop(code, None)
        if entry is None:
            return None
        if datetime.now(timezone.utc) >= entry.expires_at:
            return None
        return entry

    async def active_count(self) -> int:
        """For tests — number of unexpired live codes in the store."""
        async with self._lock:
            self._sweep_expired_locked()
            return len(self._codes)

    def _sweep_expired_locked(self) -> None:
        """Drop expired entries. Caller must hold `self._lock`."""
        now = datetime.now(timezone.utc)
        for code in [c for c, entry in self._codes.items() if entry.expires_at <= now]:
            del self._codes[code]


# Module-level singleton. Tests swap via `reset_for_testing`.
_store: PairingCodeStore | None = None


def get_store() -> PairingCodeStore:
    global _store
    if _store is None:
        _store = PairingCodeStore()
    return _store


def reset_for_testing() -> PairingCodeStore:
    global _store
    _store = PairingCodeStore()
    return _store
