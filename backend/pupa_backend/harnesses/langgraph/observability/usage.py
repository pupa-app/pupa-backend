"""Langfuse read path — aggregate token + cost totals per session (thread_id).

Companion to :mod:`pupa_backend.harnesses.langgraph.observability.tracing`, which is the *write* path. Tracing stamps
every trace with ``langfuse_session_id == thread_id``; this module reads those
traces back, aggregated, so the client can show per-thread usage.

Design notes:

- **Lazy + optional.** The ``langfuse`` package and the read client are imported
  / constructed only on first use. When tracing is off (credentials missing or
  ``PUPA_LANGFUSE_DISABLED`` set), :func:`fetch_usage` returns ``{}`` — callers
  render ``null`` usage.
- **Batched.** A single Metrics API call aggregates *all* requested sessions,
  grouped by ``sessionId``. Per-thread payloads are tiny, so we never fan out to
  one request per thread.
- **Never raises.** Any error (network, unsupported self-host, SDK shape drift)
  is logged and swallowed → ``{}``. The endpoint degrades to ``null`` usage
  rather than 500ing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio

from pupa_backend.harnesses.langgraph.observability.tracing import langfuse_enabled, langfuse_envs_present

logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class UsageRow:
    """Aggregated totals for one session. Fields are ``None`` when unknown."""

    total_tokens: int | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class CacheRow:
    """Prompt-cache breakdown of input tokens for one session.

    ``input_total`` is every input token (fresh + cache-creation + cache-read).
    ``cache_read_pct`` is ``input_cache_read / input_total * 100`` — the share of
    input served from Anthropic's prompt cache. Fields are ``None`` when unknown.
    """

    input_total: int | None = None
    input_cache_read: int | None = None
    input_cache_creation: int | None = None
    cache_read_pct: float | None = None


# --- lazy client -----------------------------------------------------------

_CLIENT: Any | None = None
_CLIENT_TRIED = False


def _client() -> Any | None:
    """Return a cached Langfuse read client, or ``None`` if unavailable."""
    global _CLIENT, _CLIENT_TRIED
    if _CLIENT is not None:
        return _CLIENT
    if _CLIENT_TRIED:
        return None
    _CLIENT_TRIED = True

    if not (langfuse_enabled() and langfuse_envs_present()):
        return None
    try:
        from langfuse import Langfuse  # noqa: PLC0415

        _CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"),
        )
    except Exception as exc:  # noqa: BLE001 — optional dependency / config
        logger.warning("[langfuse] read client unavailable: %s", exc)
        _CLIENT = None
    return _CLIENT


# --- parsing helpers -------------------------------------------------------

def _pick(row: dict[str, Any], *needles: str) -> Any:
    """Return the first value whose key contains any of *needles* (case-insensitive)."""
    for key, val in row.items():
        low = key.lower()
        if any(n in low for n in needles):
            return val
    return None


def _to_int(v: Any) -> int | None:
    try:
        return int(round(float(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# Window covering "all of a thread's history". A year back is plenty; threads
# are short-lived per New Session.
def _window() -> tuple[str, str]:
    now = datetime.now(UTC)
    return (now - timedelta(days=365)).isoformat(), (now + timedelta(minutes=1)).isoformat()


def _build_query(session_ids: list[str]) -> str:
    frm, to = _window()
    # Token measures live on the `observations` view; cost rolls up on `traces`
    # but `observations` carries totalCost too, so one view covers all four.
    return json.dumps(
        {
            "view": "observations",
            "metrics": [
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "inputTokens", "aggregation": "sum"},
                {"measure": "outputTokens", "aggregation": "sum"},
            ],
            "dimensions": [{"field": "sessionId"}],
            "filters": [
                {
                    "column": "sessionId",
                    "operator": "any of",
                    "type": "stringOptions",
                    "value": session_ids,
                }
            ],
            "fromTimestamp": frm,
            "toTimestamp": to,
        }
    )


def _measure(row: dict[str, Any], name: str) -> Any:
    """Read a summed metric. Langfuse names columns ``sum_<measure>``; fall back
    to a fuzzy match for SDK/version drift."""
    if f"sum_{name}" in row:
        return row[f"sum_{name}"]
    return _pick(row, name.lower())


def _query_metrics(client: Any, session_ids: list[str]) -> dict[str, UsageRow]:
    """Single Metrics API call → {session_id: UsageRow}. Raises on failure."""
    resp = client.api.metrics.metrics(query=_build_query(session_ids))
    rows = getattr(resp, "data", None) or []
    out: dict[str, UsageRow] = {}
    for row in rows:
        if not isinstance(row, dict):
            row = dict(row)  # tolerate pydantic / fern model rows
        sid = _pick(row, "sessionid", "session_id")
        if sid is None:
            continue
        out[str(sid)] = UsageRow(
            total_tokens=_to_int(_measure(row, "totalTokens")),
            cost_usd=_to_float(_measure(row, "totalCost")),
            input_tokens=_to_int(_measure(row, "inputTokens")),
            output_tokens=_to_int(_measure(row, "outputTokens")),
        )
    return out


def _fetch_usage_sync(session_ids: list[str]) -> dict[str, UsageRow]:
    client = _client()
    if client is None or not session_ids:
        return {}
    try:
        return _query_metrics(client, session_ids)
    except Exception as exc:  # noqa: BLE001 — degrade to empty, never 500
        logger.warning("[langfuse] usage metrics query failed: %s", exc)
        return {}


async def fetch_usage(session_ids: list[str]) -> dict[str, UsageRow]:
    """Aggregate token + cost per session in one batched call.

    Returns a map keyed by session_id (== thread_id). Sessions with no ingested
    trace are absent from the map; callers treat absence as ``null`` usage.
    Returns ``{}`` when Langfuse is disabled or the query fails.
    """
    # client.api.* is synchronous (httpx) — offload to a worker thread.
    return await anyio.to_thread.run_sync(_fetch_usage_sync, session_ids)


# --- cache breakdown (heavy, on-demand) ------------------------------------
# The Metrics API can't aggregate the prompt-cache sub-keys, so we walk each
# session's traces → GENERATION observations and sum usage_details. This is
# O(traces × pages) per session — fine for an on-demand drill-down, not for a
# bulk dashboard paint.

def _session_cache_sync(client: Any, session_id: str) -> CacheRow | None:
    traces = client.api.trace.list(session_id=session_id, limit=100)
    tdata = getattr(traces, "data", None) or []
    read = creation = fresh = 0
    seen = False
    for t in tdata:
        tid = getattr(t, "id", None)
        if not tid:
            continue
        page = 1
        while True:
            obs = client.api.observations.get_many(trace_id=tid, limit=50, page=page)
            odata = getattr(obs, "data", None) or []
            for o in odata:
                if getattr(o, "type", "") != "GENERATION":
                    continue
                ud = getattr(o, "usage_details", None) or getattr(o, "usageDetails", None) or {}
                if not isinstance(ud, dict):
                    try:
                        ud = dict(ud)
                    except Exception:  # noqa: BLE001
                        continue
                read += int(ud.get("input_cache_read") or 0)
                creation += int(ud.get("input_cache_creation") or 0)
                fresh += int(ud.get("input") or 0)
                seen = True
            meta = getattr(obs, "meta", None)
            total_pages = getattr(meta, "total_pages", None) if meta is not None else None
            if not total_pages or page >= total_pages:
                break
            page += 1
    if not seen:
        return None
    total = read + creation + fresh
    pct = (read / total * 100.0) if total else None
    return CacheRow(
        input_total=total,
        input_cache_read=read,
        input_cache_creation=creation,
        cache_read_pct=pct,
    )


def _fetch_cache_sync(session_ids: list[str]) -> dict[str, CacheRow]:
    client = _client()
    if client is None or not session_ids:
        return {}
    out: dict[str, CacheRow] = {}
    for sid in session_ids:
        try:
            row = _session_cache_sync(client, sid)
        except Exception as exc:  # noqa: BLE001 — degrade per session
            logger.warning("[langfuse] cache walk failed for %s: %s", sid, exc)
            continue
        if row is not None:
            out[sid] = row
    return out


async def fetch_cache(session_ids: list[str]) -> dict[str, CacheRow]:
    """Per-session prompt-cache breakdown. Heavier than :func:`fetch_usage`
    (walks observations) — call on demand, not for a bulk dashboard paint.

    Returns a map keyed by session_id; sessions with no generations are absent.
    ``{}`` when Langfuse is disabled.
    """
    return await anyio.to_thread.run_sync(_fetch_cache_sync, session_ids)
