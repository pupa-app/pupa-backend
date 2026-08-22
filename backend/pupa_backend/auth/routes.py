"""`GET /auth/config` — the always-public probe that tells clients which auth
method (if any) this backend requires.

Drives the per-backend status badge in the iOS Settings list.

Response shape:
- `authRequired: bool` — true by default (auth is required out of the box).
  Only `PUPA_AUTH_DISABLED=1` flips it to false (the same-laptop dev
  opt-out for `make backend` + `make mac-demo`).
- `methods: [str]` — list of accepted credential mechanisms. `"api_key"`
  when `PUPA_API_KEY` is set (server-side bootstrap); future entries
  will include `"pairing"` and, later, `"apple"`. Empty when the only
  valid credentials are paired-device tokens.
- `version: str` — backend package version, useful for client-side
  compatibility checks.
"""

import os
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from pupa_backend.version import backend_version

from .devices import DEFAULT_SCOPES, get_store
from .pairing import get_store as get_pair_store
from .scopes import require_api_key

router = APIRouter()


def _public_paths() -> set[str]:
    """Paths the auth middleware lets through without a Bearer header. The
    pairing exchange route is here — the bootstrap code IS the credential at
    that point, and the device doesn't have a token yet.

    Mirrored in `middleware._is_public` so a single source of truth is overly
    elaborate for two paths; we just keep them in sync by convention.
    """
    return {"/auth/config", "/auth/pair"}


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no")


@router.get("/config")
async def auth_config() -> dict:
    # Mirrors the middleware: auth is required by default; only the explicit
    # opt-out env var turns it off.
    auth_disabled = _truthy(os.getenv("PUPA_AUTH_DISABLED"))
    auth_required = not auth_disabled

    methods: list[str] = []
    if auth_required and bool(os.getenv("PUPA_API_KEY")):
        methods.append("api_key")

    return {
        "authRequired": auth_required,
        "methods": methods,
        "version": backend_version(),
    }


@router.get("/devices", dependencies=[Depends(require_api_key())])
async def list_devices() -> list[dict]:
    """List paired devices (no tokens in the response).

    Operator-only (``PUPA_API_KEY``). Paired devices cannot list siblings —
    device management is a privileged surface, and exposing it to any
    bearer would let a stolen device enumerate the others.
    """
    devices = await get_store().list_active()
    return [
        {
            "id": d.id,
            "label": d.label,
            "scopes": list(d.scopes),
            "createdAt": d.created_at,
            "expiresAt": d.expires_at,
        }
        for d in devices
    ]


@router.delete("/devices/{device_id}", dependencies=[Depends(require_api_key())])
async def revoke_device(device_id: str) -> dict:
    revoked = await get_store().revoke(device_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="device not found or already revoked")
    return {"revoked": device_id}


# ---------------------------------------------------------------------------
# Pair-once bootstrap
# ---------------------------------------------------------------------------


class _PairBeginRequest(BaseModel):
    label: str | None = Field(None, description="Suggested device label, e.g. 'User's iPhone'.")
    scopes: list[str] | None = Field(None, description="Scopes to grant; defaults to the full set.")
    # Short-lived: caller-controlled lifetime of the 8-char bootstrap code
    # itself (i.e. how long until it can no longer be redeemed). Defaults to
    # 5 minutes (DEFAULT_TTL in pairing.py).
    codeTtlSeconds: int | None = Field(
        None, gt=0, le=86400,
        description="Lifetime of the bootstrap code in seconds (1..86400). Default 300.",
    )
    # Long-lived: lifetime of the device token *minted by* redeeming this
    # code. None → token never expires (LAN default). Cloud deployments
    # should set a finite value (e.g. 30 days = 2592000).
    deviceTokenTtlSeconds: int | None = Field(
        None, gt=0,
        description="Lifetime of the issued device token in seconds. Default: never expires.",
    )

    @field_validator("scopes")
    @classmethod
    def _known_scopes(cls, v: list[str] | None) -> list[str] | None:
        """Reject scope names outside `DEFAULT_SCOPES`. A typo would otherwise
        mint a device whose scope matches no route — silently useless, and
        indistinguishable from a downgrade attempt."""
        if v is None:
            return v
        unknown = sorted(set(v) - set(DEFAULT_SCOPES))
        if unknown:
            raise ValueError(f"unknown scopes: {', '.join(unknown)}")
        return v


class _PairExchangeRequest(BaseModel):
    code: str
    label: str = Field(min_length=1, max_length=200)


@router.post("/pair/begin", dependencies=[Depends(require_api_key())])
async def pair_begin(body: _PairBeginRequest | None = None) -> dict:
    """Operator-side: mint a short-lived bootstrap code. **Operator-only** —
    `PUPA_API_KEY` is required, so `PUPA_API_KEY` must stay set for as long as
    you want to pair devices.

    A paired-device token is deliberately *not* enough: minting is how device
    tokens come into existence, so letting one device mint another would make a
    leaked token outlive the revocation of the device it was issued to.
    """
    from .pairing import DEFAULT_TTL

    payload = body or _PairBeginRequest()
    # `is not None`, not truthiness: an explicit `[]` asks for a device with
    # no scopes at all, which is the opposite of the default set.
    scopes = payload.scopes if payload.scopes is not None else list(DEFAULT_SCOPES)
    code_ttl = (
        timedelta(seconds=payload.codeTtlSeconds)
        if payload.codeTtlSeconds is not None
        else DEFAULT_TTL
    )
    device_ttl = (
        timedelta(seconds=payload.deviceTokenTtlSeconds)
        if payload.deviceTokenTtlSeconds is not None
        else None
    )
    entry = await get_pair_store().mint(
        suggested_label=payload.label,
        scopes=scopes,
        ttl=code_ttl,
        device_token_ttl=device_ttl,
    )
    return {
        "code": entry.code,
        "expiresAt": entry.expires_at_iso,
        "scopes": list(entry.scopes),
        "suggestedLabel": entry.suggested_label,
        "deviceTokenTtlSeconds": payload.deviceTokenTtlSeconds,
    }


@router.post("/pair")
async def pair_exchange(body: _PairExchangeRequest) -> dict:
    """Device-side: redeem a bootstrap code for a permanent device token.
    *Public* — the code IS the credential at this point. The device exists
    on the same LAN/tunnel as the operator who minted the code.
    """
    entry = await get_pair_store().consume(body.code)
    if entry is None:
        raise HTTPException(status_code=404, detail="invalid or expired pairing code")
    label = body.label.strip() or entry.suggested_label or "Unnamed device"
    device_id, token = await get_store().issue(
        label=label, scopes=entry.scopes, ttl=entry.device_token_ttl,
    )
    return {
        "deviceId": device_id,
        "token": token,
        "label": label,
        "scopes": list(entry.scopes),
    }
