"""Langfuse v3 tracing helpers for the Pupa AG-UI endpoint.

Tracing is **opt-in** and **lazy** — the langfuse package is imported only
when tracing is actually used, so the dependency stays optional at runtime.

## Activation (opt-out)

Tracing is **on by default** whenever the Langfuse credentials are present:

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_BASE_URL=http://localhost:3000   # defaults to cloud if unset

No flag is needed to turn it on. To turn it **off** while keeping the
credentials in the environment, set ``PUPA_LANGFUSE_DISABLED=1``.

Tracing is entirely a server-side concern: it configures itself from the
environment plus the AG-UI identifiers every request already carries, so
clients neither know nor say anything about Langfuse.

    trace_id   ← AG-UI run_id when it is a UUID, else a random one
    session_id ← AG-UI thread_id

## Version tracking

Every trace is stamped with the backend build: the ``release`` field (via the
``LANGFUSE_RELEASE`` env var, set once in app startup to ``pupa-backend@<ver>``)
and a ``backend:<ver>`` tag.

## No spurious traces

``CallbackHandler`` is constructed with ``update_trace=False`` so it never
auto-emits a root trace on construction — it only attaches child spans to the
``trace_id`` we supply.  Session and tags reach Langfuse via LangGraph
metadata keys (``langfuse_session_id`` / ``langfuse_tags``).
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from pupa_backend.version import backend_version

if TYPE_CHECKING:
    # Only for type-checking; never executed at import time.
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # noqa: F401

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in _TRUTHY


def langfuse_envs_present() -> bool:
    """Return True when the minimum Langfuse credentials are in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_disabled() -> bool:
    """Return True when PUPA_LANGFUSE_DISABLED is set to a truthy value (opt-out)."""
    return _truthy(os.getenv("PUPA_LANGFUSE_DISABLED"))


def langfuse_enabled() -> bool:
    """Return True when Langfuse tracing should run.

    Opt-out model: tracing is **on by default** whenever the credentials are
    present, unless ``PUPA_LANGFUSE_DISABLED`` is truthy. No flag is needed to
    turn it on.
    """
    return langfuse_envs_present() and not langfuse_disabled()


def _random_trace_id() -> str:
    """Generate a random 32-char lowercase hex trace_id (no hyphens)."""
    return uuid.uuid4().hex


def _normalise_uuid(v: Any) -> str:
    """Validate *v* is a UUID string and return it as 32 lowercase hex chars."""
    try:
        uuid.UUID(str(v))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"trace_id must be a valid UUID string, got: {v!r}") from exc
    return str(v).replace("-", "").lower()


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

def _import_callback_handler() -> type:
    """Import LangfuseCallbackHandler lazily — raises ImportError if not installed."""
    try:
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415
        return CallbackHandler
    except ImportError as exc:
        raise ImportError(
            "langfuse is not installed. "
            "Add 'langfuse>=3.0.0,<4.0.0' to your dependencies or "
            "run: uv pip install 'langfuse>=3.0.0,<4.0.0'"
        ) from exc


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_langfuse_config(
    *,
    thread_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return a LangGraph config fragment that wires up Langfuse tracing.

    ``update_trace=False`` ensures no spurious root trace is emitted on
    construction — the handler only records spans under the supplied trace_id.

    Args:
        thread_id: AG-UI thread_id — becomes the Langfuse session_id.
        run_id:    AG-UI run_id — becomes the trace_id when it is a UUID.

    Returns:
        Dict with ``callbacks`` and ``metadata`` ready to merge into self.config.
    """
    CallbackHandler = _import_callback_handler()

    if run_id:
        try:
            trace_id = _normalise_uuid(run_id)
        except ValueError:
            trace_id = _random_trace_id()
            logger.debug("[langfuse] run_id %r is not a valid UUID; generated trace_id=%s", run_id, trace_id)
    else:
        trace_id = _random_trace_id()
        logger.debug("[langfuse] no trace_id or run_id supplied; generated trace_id=%s", trace_id)

    logger.debug(
        "[langfuse] tracing request — trace_id=%s session_id=%s", trace_id, thread_id,
    )

    handler = CallbackHandler(
        trace_context={"trace_id": trace_id},
        update_trace=False,  # never auto-emit a root trace
    )

    # The version tag reaches the trace level via the `langfuse_tags` key; the
    # raw value is also kept as root-observation metadata for searching. The
    # `release` field carries it too (set via LANGFUSE_RELEASE in app startup).
    bver = backend_version()
    metadata: dict[str, Any] = {
        "langfuse_session_id": thread_id,
        "backend_version": bver,
        "langfuse_tags": [f"backend:{bver}"],
    }

    return {"callbacks": [handler], "metadata": metadata}


# ---------------------------------------------------------------------------
# Convenience: resolve config from the environment
# ---------------------------------------------------------------------------

def resolve_langfuse_config(
    *,
    thread_id: str,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a Langfuse config fragment, or None when tracing is off.

    Tracing runs when the credentials are present and ``PUPA_LANGFUSE_DISABLED``
    is not set; otherwise this returns None and costs nothing. Single call-site
    in the harness, so the run path stays a simple if/else.
    """
    if not langfuse_enabled():
        return None
    try:
        return build_langfuse_config(thread_id=thread_id, run_id=run_id)
    except ImportError as exc:
        logger.warning("[langfuse] %s — tracing skipped.", exc)
        return None
