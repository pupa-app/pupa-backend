"""FastAPI routes over the deepagents harness's LangGraph checkpointer.

Mounted under /db alongside that harness, and only when it is enabled —
another loop's threads are not visible here. The checkpointer is read
from ``app.state.checkpointer``, set by the FastAPI lifespan in app.py.

No assistant-running routes here — this module exposes only persistence I/O.

Authorization (see :mod:`auth.scopes`): every route requires the ``agent``
scope (or the operator key). Devices need it to load their own chat history;
thread ids are random per session, so collisions across devices are
practically impossible.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langgraph.checkpoint.base import BaseCheckpointSaver

from pupa_backend.auth import require_scope
from pupa_backend.harnesses.langgraph.observability.usage import fetch_cache, fetch_usage

from .connection import thread_config
from .schemas import (
    ThreadCacheBatchResponse,
    ThreadCacheUsage,
    ThreadUsage,
    ThreadUsageBatchRequest,
    ThreadUsageBatchResponse,
    ToolCallEntry,
    TranscriptMessage,
)

router = APIRouter()

_AGENT = [Depends(require_scope("agent"))]


def _checkpointer(request: Request) -> BaseCheckpointSaver:
    saver = getattr(request.app.state, "checkpointer", None)
    if saver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer is not initialised.",
        )
    return saver


async def _latest_checkpoint(request: Request, thread_id: str):
    return await _checkpointer(request).aget_tuple(thread_config(thread_id))


@router.get(
    "/threads/{thread_id}/messages",
    response_model=list[TranscriptMessage],
    tags=["Deep Agents threads"],
    dependencies=_AGENT,
)
async def get_thread_messages(thread_id: str, request: Request) -> list[TranscriptMessage]:
    """Return a normalized, ordered transcript for a thread.

    Pulls the latest checkpoint and extracts the ``messages`` channel.
    Returns ``[]`` when the thread has no checkpoint yet rather than 404,
    so callers can distinguish "unknown thread" from "thread with no history"
    at the application layer if needed.
    """
    tup = await _latest_checkpoint(request, thread_id)
    if tup is None:
        return []
    raw_messages = tup.checkpoint.get("channel_values", {}).get("messages", [])
    return _normalize_messages(raw_messages)


def _normalize_messages(raw: list) -> list[TranscriptMessage]:
    """Convert a mix of LangChain message objects or dicts into TranscriptMessage."""
    from langchain_core.messages import messages_to_dict

    result: list[TranscriptMessage] = []
    for item in raw:
        if isinstance(item, dict):
            d = item
        else:
            # LangChain message object — serialize then work with the dict form
            d = messages_to_dict([item])[0]

        # messages_to_dict wraps content inside {"type": ..., "data": {...}};
        # raw dicts from the checkpointer may already have the flat form.
        if "data" in d:
            data = d["data"]
            raw_role = d.get("type", "")
        else:
            data = d
            raw_role = d.get("type", d.get("role", ""))

        # LangChain roles: "human" / "ai" / "tool"
        role = raw_role

        content_raw = data.get("content", "")
        content = content_raw if isinstance(content_raw, str) else ""

        tool_calls = [
            ToolCallEntry(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                args=tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
            )
            for tc in (data.get("tool_calls") or [])
        ]

        result.append(
            TranscriptMessage(
                id=data.get("id"),
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=data.get("tool_call_id"),
            )
        )
    return result


@router.delete("/threads/{thread_id}", tags=["Deep Agents threads"], dependencies=_AGENT)
async def delete_thread(thread_id: str, request: Request) -> dict:
    await _checkpointer(request).adelete_thread(thread_id)
    return {"thread_id": thread_id, "deleted": True}


# --- Usage (Langfuse-backed token + cost) ----------------------------------
# Cached per thread, keyed by latest checkpoint_id ("fingerprint"). We only hit
# Langfuse for threads whose fingerprint changed since last fetch (= a new turn)
# or whose cached value lapsed the TTL grace window (covers ingestion lag right
# after a turn). Plain-dict cache, mirroring the _GRAPH_CACHE idiom in agent.py.

_USAGE_CACHE: dict[str, ThreadUsage] = {}
_USAGE_FETCHED_AT: dict[str, float] = {}
_USAGE_TTL_SECONDS = 30.0


def _usage_fresh(thread_id: str) -> bool:
    ts = _USAGE_FETCHED_AT.get(thread_id)
    return ts is not None and (time.monotonic() - ts) < _USAGE_TTL_SECONDS


async def _thread_fingerprint(request: Request, thread_id: str) -> str | None:
    tup = await _latest_checkpoint(request, thread_id)
    if tup is None:
        return None
    return tup.config.get("configurable", {}).get("checkpoint_id")


@router.post(
    "/threads/usage",
    response_model=ThreadUsageBatchResponse,
    tags=["Deep Agents threads"],
    dependencies=_AGENT,
)
async def get_threads_usage(
    body: ThreadUsageBatchRequest,
    request: Request,
) -> ThreadUsageBatchResponse:
    """Batch token + cost totals for many threads in one call.

    Returns a map keyed by thread_id. ``total_tokens`` / ``cost_usd`` are
    ``null`` when Langfuse is disabled or a trace hasn't ingested yet.
    """
    out: dict[str, ThreadUsage] = {}
    stale: list[str] = []

    for tid in body.thread_ids:
        fingerprint = await _thread_fingerprint(request, tid)
        cached = _USAGE_CACHE.get(tid)
        if cached is not None and cached.fingerprint == fingerprint and _usage_fresh(tid):
            out[tid] = cached  # unchanged turn → serve cache, no Langfuse call
        else:
            out[tid] = ThreadUsage(thread_id=tid, fingerprint=fingerprint)
            stale.append(tid)

    if stale:
        fetched = await fetch_usage(stale)  # {} when Langfuse disabled
        now = time.monotonic()
        for tid in stale:
            row = fetched.get(tid)
            entry = out[tid]
            if row is not None:
                entry.total_tokens = row.total_tokens
                entry.cost_usd = row.cost_usd
                entry.input_tokens = row.input_tokens
                entry.output_tokens = row.output_tokens
            _USAGE_CACHE[tid] = entry
            _USAGE_FETCHED_AT[tid] = now

    return ThreadUsageBatchResponse(usage=out)


# --- Cache breakdown (on-demand, heavier) ----------------------------------
# Prompt-cache % can't be aggregated by the Metrics API, so this walks each
# session's generations — call it on demand (e.g. expanding one agent), not on
# every dashboard paint. Same fingerprint + TTL cache as plain usage.

_CACHE_USAGE: dict[str, ThreadCacheUsage] = {}
_CACHE_FETCHED_AT: dict[str, float] = {}


def _cache_fresh(thread_id: str) -> bool:
    ts = _CACHE_FETCHED_AT.get(thread_id)
    return ts is not None and (time.monotonic() - ts) < _USAGE_TTL_SECONDS


@router.post(
    "/threads/usage/cache",
    response_model=ThreadCacheBatchResponse,
    tags=["Deep Agents threads"],
    dependencies=_AGENT,
)
async def get_threads_cache(
    body: ThreadUsageBatchRequest,
    request: Request,
) -> ThreadCacheBatchResponse:
    """Batch prompt-cache breakdown for threads. Heavier than ``/usage`` (walks
    Langfuse observations) — intended for on-demand drill-down, not bulk paint.
    """
    out: dict[str, ThreadCacheUsage] = {}
    stale: list[str] = []

    for tid in body.thread_ids:
        fingerprint = await _thread_fingerprint(request, tid)
        cached = _CACHE_USAGE.get(tid)
        if cached is not None and cached.fingerprint == fingerprint and _cache_fresh(tid):
            out[tid] = cached
        else:
            out[tid] = ThreadCacheUsage(thread_id=tid, fingerprint=fingerprint)
            stale.append(tid)

    if stale:
        fetched = await fetch_cache(stale)  # {} when Langfuse disabled
        now = time.monotonic()
        for tid in stale:
            row = fetched.get(tid)
            entry = out[tid]
            if row is not None:
                entry.input_total = row.input_total
                entry.input_cache_read = row.input_cache_read
                entry.input_cache_creation = row.input_cache_creation
                entry.cache_read_pct = row.cache_read_pct
            _CACHE_USAGE[tid] = entry
            _CACHE_FETCHED_AT[tid] = now

    return ThreadCacheBatchResponse(usage=out)
