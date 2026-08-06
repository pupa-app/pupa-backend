"""Tests for Langfuse trace wiring in ``observability/tracing.py``.

Tracing configures itself from the environment plus the AG-UI identifiers the
request already carries: the backend build is stamped on every trace
(``backend:<ver>`` tag + ``backend_version`` metadata; the ``release`` field is
set separately via ``LANGFUSE_RELEASE`` in app startup), the ``session_id``
comes from the ``thread_id``, and the ``trace_id`` from the ``run_id``.

The ``CallbackHandler`` is monkeypatched to a dummy so these tests need no
langfuse credentials and never construct a real client.
"""

import uuid

import pupa_backend.harnesses.langgraph.observability.tracing as tracing
from pupa_backend.harnesses.langgraph.observability.tracing import (
    backend_version,
    build_langfuse_config,
    resolve_langfuse_config,
)


class _DummyHandler:
    """Stand-in for ``langfuse.langchain.CallbackHandler`` — accepts the kwargs
    ``build_langfuse_config`` passes and does nothing."""

    def __init__(self, *, trace_context=None, update_trace=False):
        self.trace_context = trace_context
        self.update_trace = update_trace


def _patch_handler(monkeypatch):
    monkeypatch.setattr(tracing, "_import_callback_handler", lambda: _DummyHandler)


def _enable(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv("PUPA_LANGFUSE_DISABLED", raising=False)


def test_backend_version_is_stamped(monkeypatch):
    _patch_handler(monkeypatch)

    cfg = build_langfuse_config(thread_id="t")
    meta = cfg["metadata"]
    bver = backend_version()

    assert meta["langfuse_tags"] == [f"backend:{bver}"]
    assert meta["backend_version"] == bver


def test_session_id_is_the_thread_id(monkeypatch):
    _patch_handler(monkeypatch)

    cfg = build_langfuse_config(thread_id="thread-42")
    assert cfg["metadata"]["langfuse_session_id"] == "thread-42"


def test_trace_id_comes_from_a_uuid_run_id(monkeypatch):
    _patch_handler(monkeypatch)

    run_id = str(uuid.uuid4())
    cfg = build_langfuse_config(thread_id="t", run_id=run_id)

    handler = cfg["callbacks"][0]
    assert handler.trace_context == {"trace_id": run_id.replace("-", "").lower()}
    assert handler.update_trace is False  # never auto-emit a root trace


def test_non_uuid_run_id_falls_back_to_a_random_trace_id(monkeypatch):
    """A run_id that isn't a UUID must not break the run — trace it anyway."""
    _patch_handler(monkeypatch)

    cfg = build_langfuse_config(thread_id="t", run_id="not-a-uuid")
    trace_id = cfg["callbacks"][0].trace_context["trace_id"]

    assert trace_id != "not-a-uuid"
    assert len(trace_id) == 32


def test_resolve_returns_config_when_credentials_present(monkeypatch):
    _patch_handler(monkeypatch)
    _enable(monkeypatch)

    cfg = resolve_langfuse_config(thread_id="t")
    assert cfg is not None
    assert f"backend:{backend_version()}" in cfg["metadata"]["langfuse_tags"]


def test_resolve_returns_none_without_credentials(monkeypatch):
    _patch_handler(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert resolve_langfuse_config(thread_id="t") is None


def test_disabled_flag_is_an_absolute_kill_switch(monkeypatch):
    """Credentials present but the opt-out set — nothing may reach Langfuse."""
    _patch_handler(monkeypatch)
    _enable(monkeypatch)
    monkeypatch.setenv("PUPA_LANGFUSE_DISABLED", "1")

    assert resolve_langfuse_config(thread_id="t") is None
