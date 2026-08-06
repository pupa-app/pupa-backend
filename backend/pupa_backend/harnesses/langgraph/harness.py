"""LangGraph agent harness — the default `POST /` AG-UI handler.

Extracted from `app.py` so the harness registry (`harnesses.py`) can mount it
without importing `app` (which would be circular). Holds the AG-UI agent
subclass that adds per-request Langfuse tracing + per-request model selection,
and the `DeepAgentsHarness` adapter the registry drives.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Any

from ag_ui.core.events import EventType, RunErrorEvent
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from pydantic import BaseModel, ConfigDict, Field

from pupa_backend.harnesses.langgraph.observability.tracing import resolve_langfuse_config

logger = logging.getLogger("uvicorn.error")


class LLMParams(BaseModel):
    """Per-request LLM selection forwarded via ``forwardedProps["llm"]``.

    Both fields must be set together — partial input is rejected at the call
    site (``_resolve_per_request_graph``) so the iOS user sees a clear toast
    rather than the request silently falling back to the env default.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(
        default=None,
        description="One of 'bedrock', 'anthropic', 'openai_compatible'.",
    )
    model: str | None = Field(
        default=None,
        description="Logical model id (e.g. 'claude-sonnet-4-6'). Mapped to a "
                    "provider-specific id via agent.MODEL_REGISTRY.",
    )


class CustomLangGraphAGUIAgent(LangGraphAGUIAgent):
    """LangGraphAGUIAgent with optional per-request Langfuse tracing AND per-request
    graph (model) selection.

    **Tracing** activates whenever Langfuse credentials are present and
    ``PUPA_LANGFUSE_DISABLED`` is unset (opt-out: on by default). It is a
    server-side concern only — the ``trace_id`` comes from the AG-UI
    ``run_id`` (or a random UUID when that is not one) and the ``session_id``
    from the ``thread_id``.

    **Per-request model selection** activates when the client sends
    ``forwarded_props["llm"] = {"provider": ..., "model": ...}``. The matching
    cached graph from ``agent.get_graph`` replaces ``self.graph`` for the
    duration of the request and is restored in ``finally``. If the (provider,
    model) pair is unknown or the chosen provider lacks creds, a
    ``RunErrorEvent`` is yielded and the run ends cleanly — no exception
    propagates past the AG-UI stream boundary.

    Both ``self.config`` and ``self.graph`` mutations are scoped to the request
    by ``_request_overrides``, so concurrent requests on the same agent instance
    don't bleed state into each other.
    """

    async def _handle_stream_events(  # type: ignore[override]
        self, input: Any
    ) -> AsyncGenerator:

        # --- per-request graph (model) selection ----------------------------
        # Note: ag_ui_langgraph's LangGraphAGUIAgent.run() normalises camelCase
        # keys in forwarded_props to snake_case before calling this method, so
        # the iOS payload `forwardedProps.llm` lands here as `forwarded_props["llm"]`.
        try:
            per_request_graph = _resolve_per_request_graph(input.forwarded_props)
        except _PerRequestModelError as exc:
            yield self._dispatch_event(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(exc),
                    code="llm_unavailable",
                )
            )
            return

        # --- per-request Langfuse tracing -----------------------------------
        extra_config = resolve_langfuse_config(
            thread_id=input.thread_id or "",
            run_id=getattr(input, "run_id", None),
        )

        with self._request_overrides(graph=per_request_graph, config=extra_config):
            async for event in super()._handle_stream_events(input):
                yield event

    @contextmanager
    def _request_overrides(
        self, graph: Any | None = None, config: dict | None = None
    ) -> Iterator[None]:
        """Swap in a per-request ``graph`` / ``config``, restoring both on exit.

        A ``None`` argument means "leave that attribute alone"; ``config`` is
        merged over the instance default rather than replacing it. Restoring in
        ``finally`` keeps concurrent requests on this shared agent instance from
        inheriting each other's overrides.
        """
        saved_graph = self.graph
        saved_config = self.config or {}
        try:
            if graph is not None:
                self.graph = graph
            if config is not None:
                self.config = {**saved_config, **config}
            yield
        finally:
            self.graph = saved_graph
            self.config = saved_config


class _PerRequestModelError(Exception):
    """Raised inside `_resolve_per_request_graph` so the AG-UI handler can yield
    a `RunErrorEvent` with the same message."""


def _resolve_per_request_graph(forwarded_props: Any):
    """Return the cached graph for the (provider, model) in `forwarded_props["llm"]`,
    or `None` if no per-request model was requested.

    Bad shapes are logged-and-skipped (treated as no override) so a malformed
    `llm` block from an old client doesn't kill the request. Real construction
    failures (unknown model, missing creds) raise `_PerRequestModelError` so
    the caller can surface them as a `RunErrorEvent`.
    """
    from pupa_backend.harnesses.langgraph.agent import (
        MissingCredentialsError,
        UnknownModelError,
        get_graph,
    )

    props = forwarded_props or {}
    raw = props.get("llm") if isinstance(props, dict) else None
    if raw is None:
        return None
    try:
        params = LLMParams(**raw)
    except Exception as exc:
        logger.warning("[llm] invalid llm block in forwarded_props: %s — using default model", exc)
        return None
    # Both fields must be set; either-only is a client bug worth surfacing.
    if params.provider is None and params.model is None:
        return None
    if params.provider is None or params.model is None:
        raise _PerRequestModelError(
            f"forwardedProps.llm requires both 'provider' and 'model' "
            f"(got provider={params.provider!r}, model={params.model!r})."
        )
    try:
        return get_graph(provider=params.provider, model_id=params.model)
    except (UnknownModelError, MissingCredentialsError) as exc:
        raise _PerRequestModelError(str(exc)) from exc


class DeepAgentsHarness:
    """The deepagents loop as a discoverable, mountable harness.

    `prepare()` builds the shared per-(provider,model) graph deps once; `register`
    mounts an AG-UI endpoint at the given path (may be called for both
    `/harnesses/deepagents` and the `/` alias — the graph build is idempotent).
    """

    id = "deepagents"
    label = "Deep Agents"

    def register(self, app: Any, path: str, deps: Any) -> None:
        from pupa_backend.harnesses.langgraph.agent import _GRAPH_CACHE, build_graph, recursion_limit, register_graph_deps

        if (None, None) not in _GRAPH_CACHE:
            # First mount: stash persistence handles so per-request
            # `get_graph(provider, model)` can lazily build per-(provider, model)
            # graphs sharing the same checkpointer/store/mcp, and pre-build the
            # default graph so its cost is paid at startup, not the first request.
            register_graph_deps(checkpointer=deps.checkpointer, store=deps.store, mcp=deps.mcp)
            _GRAPH_CACHE[(None, None)] = build_graph(
                checkpointer=deps.checkpointer, store=deps.store, mcp=deps.mcp
            )
        graph = _GRAPH_CACHE[(None, None)]
        agent = CustomLangGraphAGUIAgent(
            name="pupa",
            description="Pupa A2UI agent",
            graph=graph,
            config={"recursion_limit": recursion_limit()},
        )
        add_langgraph_fastapi_endpoint(app=app, agent=agent, path=path)
        logger.info("deepagents harness active on POST %s", path)

    def models(self) -> list[dict]:
        from pupa_backend.harnesses.langgraph.agent import MODEL_REGISTRY

        return [
            {"provider": provider, "modelId": model_id, "label": params["label"]}
            for (provider, model_id), params in MODEL_REGISTRY.items()
        ]

    def tools(self) -> list[dict]:
        from pupa_backend.harnesses.langgraph.backend_tools import all_specs

        return [
            {
                "name": spec.name,
                "description": spec.description,
                "enabledByEnv": spec.enabled_by_env,
            }
            for spec in all_specs()
        ]

    def permission_schema(self) -> list[dict]:
        # Backend-tool mute list + the shell-approval bypass. Both keys are read
        # verbatim from RunAgentInput.state by ToolGatingMiddleware /
        # ShellApprovalMiddleware — do not rename without updating those.
        return [
            {
                "key": "disabled_tools",
                "type": "toolset",
                "label": "Backend tools",
            },
            {
                "key": "shell_approval_disabled",
                "type": "bool",
                "label": "Skip shell-command approval",
                "default": False,
            },
        ]
