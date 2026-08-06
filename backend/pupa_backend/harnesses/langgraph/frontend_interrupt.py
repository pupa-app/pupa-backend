"""Batched-interrupt routing for client-side (frontend) tools.

Subclasses ``CopilotKitMiddleware`` and overrides ``after_model`` so that
when the model emits ``tool_calls`` whose names match frontend descriptors
forwarded by the client (``state["copilotkit"]["actions"]``), the graph
pauses on a single batched ``interrupt(...)`` carrying every frontend
call. ``ag_ui_langgraph`` surfaces that pause to the client as
``CustomEvent(name="on_interrupt", value={"frontend_tool_calls": [...]})``;
the client dispatches each call locally and POSTs
``forwardedProps.command.resume = {"tool_results": [{"toolCallId", "content"}, ...]}``
to re-enter the graph. On resume, this middleware appends one
``ToolMessage(tool_call_id=…, content=…)`` per resumed result and routes
back to the model (or, if backend tool_calls share the turn, lets the
standard conditional edge route through ``ToolNode`` first).

The frontend ``tool_calls`` remain on the AIMessage throughout — that
keeps each new ``ToolMessage`` paired with a matching ``tool_call_id``
so the parent class's ``_fix_messages_for_bedrock`` doesn't erase them
as orphans, and the persisted history reflects what the model actually
emitted. The built-in ``ToolNode`` filters tool_calls by its own
registry before running, so the frontend calls are skipped there.

**Standalone usability.** This middleware works with or without
backend tools. ``@hook_config(can_jump_to=["model"])`` on
``after_model`` (combined with ``{"jump_to": "model"}`` in the
returned state for the all-frontend case) installs the conditional
edge that re-enters the model after resume even when ``create_agent``
was called with ``tools=[]``. Mixed FE+BE turns inherently need a
``tool_node``, since the backend tools have to execute somewhere. See
``langchain/agents/factory.py::_add_middleware_edge`` for the
underlying mechanism.
"""



import json
from typing import Any

from copilotkit import CopilotKitMiddleware
from copilotkit.copilotkit_lg_middleware import StateSchema
from langchain.agents.middleware import hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from pupa_backend.agui.tool_results import parse_tool_results


class CustomCopilotKitMiddleware(CopilotKitMiddleware):
    """``after_model`` hook that batch-interrupts on frontend tool calls."""

    @property
    def name(self) -> str:
        return "CustomCopilotKitMiddleware"

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: StateSchema,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        frontend_tools = state.get("copilotkit", {}).get("actions", [])
        if not frontend_tools:
            return None

        frontend_tool_names = {
            t.get("function", {}).get("name") or t.get("name")
            for t in frontend_tools
        }

        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        tool_calls = getattr(last_message, "tool_calls", None) or []
        if not tool_calls:
            return None

        frontend_calls = [c for c in tool_calls if c.get("name") in frontend_tool_names]
        backend_calls = [c for c in tool_calls if c.get("name") not in frontend_tool_names]

        if not frontend_calls:
            return None

        # Pause the graph with every frontend call batched into one
        # interrupt. On resume, ``payload`` carries the per-call results
        # the client produced.
        payload = interrupt({
            "frontend_tool_calls": [
                {"id": c.get("id"), "name": c.get("name"), "args": c.get("args") or {}}
                for c in frontend_calls
            ]
        })

        results = parse_tool_results(payload)
        results_by_id = {r["toolCallId"]: r["content"] for r in results}
        tools_after_round = _parse_tools_after_round(payload)

        # Build a ToolMessage for every frontend call. If the client
        # dropped a result (cancel, crash, etc.), synthesise a placeholder
        # so the AIMessage's tool_calls stay paired with ToolMessages and
        # Bedrock doesn't reject the conversation.
        new_tool_messages: list[ToolMessage] = []
        for call in frontend_calls:
            call_id = call.get("id")
            content = results_by_id.get(call_id)
            if content is None:
                content = json.dumps({"ok": False, "error": "missing_tool_result"})
            new_tool_messages.append(
                ToolMessage(tool_call_id=call_id, content=str(content))
            )

        # Keep the frontend tool_calls on the AIMessage so each newly
        # appended ``ToolMessage`` stays paired with a matching
        # ``tool_call_id`` — ``_fix_messages_for_bedrock`` (inherited
        # from the parent class) strips ToolMessages whose id isn't
        # present in any AIMessage's ``tool_calls``, which would
        # otherwise erase the frontend results we just received. The
        # built-in ``ToolNode`` filters by its own registry before
        # running, so leaving the frontend calls in place doesn't cause
        # it to try executing them.
        # Routing: ``_make_model_to_tools_edge`` (factory.py:1718) is
        # the conditional edge that runs after this hook when
        # ``tool_node`` exists. It honours ``state["jump_to"]`` first,
        # then falls back to checking pending tool_calls.
        #
        # - **All-frontend turn**: no backend calls left to run. We
        #   need to re-enter the model so it can react to the FE
        #   results. Setting ``jump_to: "model"`` covers BOTH the
        #   ``tool_node`` exists case AND the no-tool_node test case
        #   (where the default edge would otherwise route to END,
        #   silently stopping the run after every resume).
        # - **Mixed turn**: backend calls share the AIMessage. We must
        #   NOT set ``jump_to`` here — the default edge correctly
        #   ``Send``s only the pending backend calls to ToolNode (FE
        #   ones now have matching ToolMessages, so they're filtered
        #   out). Skipping ToolNode here would leave the backend
        #   tool_calls unanswered → model re-emits them → loop.
        update: dict[str, Any] = {"messages": new_tool_messages}
        if not backend_calls:
            update["jump_to"] = "model"
        # Mid-turn tool-surface refresh. The iOS client recomputes its
        # advertised tool list on every round, including the resume POST
        # that lands here (e.g. an `addComponent(kind:"checklist")` call
        # in this same round widened the surface to include the
        # `renderChecklist` family). ``ag_ui_langgraph`` builds the
        # would-be-fresh state from ``RunAgentInput.tools`` but on the
        # resume branch (``Command(resume=...)``) it never flushes that
        # state into the graph — so without this rewrite the next model
        # call would read the checkpointed (stale) ``copilotkit.actions``
        # and the model would never see the newly-unlocked tools.
        # ``CopilotKitMiddleware.awrap_model_call`` reads from
        # ``state["copilotkit"]["actions"]`` to merge frontend tools
        # into ``request.tools``, so writing the new list there is
        # sufficient to refresh the bound surface on the very next
        # model call inside this same turn.
        if tools_after_round is not None:
            existing_copilotkit = state.get("copilotkit", {}) or {}
            update["copilotkit"] = {
                **existing_copilotkit,
                "actions": tools_after_round,
            }
        return update

    async def aafter_model(
        self,
        state: StateSchema,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def _parse_tools_after_round(payload: Any) -> list[dict] | None:
    """Pull the optional widened tool surface out of the resume payload.

    The iOS client populates ``forwardedProps.command.resume.tools_after_round``
    with the descriptor list its ``toolFilter`` produced for the next
    round, so we can refresh ``state["copilotkit"]["actions"]`` mid-turn
    (``ag_ui_langgraph.prepare_stream`` doesn't propagate
    ``RunAgentInput.tools`` on the resume branch). Returns ``None`` when
    the client omitted the field — older clients keep working with the
    checkpointed actions, just without mid-turn refresh.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("tools_after_round")
    if not isinstance(raw, list):
        return None
    cleaned: list[dict] = []
    for entry in raw:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            cleaned.append(entry)
    return cleaned


