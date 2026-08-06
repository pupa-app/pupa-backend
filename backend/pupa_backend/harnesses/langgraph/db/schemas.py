"""Pydantic v2 schemas for the /db routes."""



from typing import Any

from pydantic import BaseModel, Field


class ToolCallEntry(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class TranscriptMessage(BaseModel):
    """One normalized message from a LangGraph thread's checkpoint."""

    id: str | None = None
    role: str  # "human" | "ai" | "tool"
    content: str
    tool_calls: list[ToolCallEntry] = Field(default_factory=list)
    tool_call_id: str | None = None


class ThreadUsage(BaseModel):
    """Token + cost totals for one thread, sourced from Langfuse.

    ``total_tokens`` / ``cost_usd`` are ``None`` when Langfuse is disabled or
    the thread has no ingested trace yet. ``fingerprint`` is the latest
    checkpoint_id and changes only when the thread gets a new turn — callers
    use it to decide whether a refetch is needed.
    """

    thread_id: str
    total_tokens: int | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    fingerprint: str | None = Field(
        default=None, description="latest checkpoint_id; changes on new turn"
    )


class ThreadUsageBatchRequest(BaseModel):
    thread_ids: list[str] = Field(..., max_length=200)


class ThreadUsageBatchResponse(BaseModel):
    usage: dict[str, ThreadUsage] = Field(
        default_factory=dict, description="keyed by thread_id"
    )


class ThreadCacheUsage(BaseModel):
    """Prompt-cache breakdown of input tokens for one thread.

    ``cache_read_pct`` is the share of input tokens served from Anthropic's
    prompt cache. ``None`` fields mean Langfuse is off or no generation ingested.
    """

    thread_id: str
    input_total: int | None = None
    input_cache_read: int | None = None
    input_cache_creation: int | None = None
    cache_read_pct: float | None = None
    fingerprint: str | None = Field(
        default=None, description="latest checkpoint_id; changes on new turn"
    )


class ThreadCacheBatchResponse(BaseModel):
    usage: dict[str, ThreadCacheUsage] = Field(
        default_factory=dict, description="keyed by thread_id"
    )

