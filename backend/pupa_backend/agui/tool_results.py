"""Normalise the frontend-tool resume payload both harnesses receive.

The iOS client answers a parked frontend tool call by POSTing
``forwardedProps.command.resume = {"tool_results": [...]}``. The LangGraph
middleware and the Claude Code loop both have to turn that into a uniform
list before pairing results back to their pending calls, so the parsing
lives here rather than in either harness.
"""

import json
from typing import Any


def parse_tool_results(payload: Any) -> list[dict]:
    """Normalise the resume payload into a list of ``{toolCallId, content}``.

    ``ag_ui_langgraph`` JSON-decodes string payloads before re-entering the
    graph, so ``payload`` is usually a dict. Tolerate a bare list as a
    courtesy so callers never crash on shape drift.
    """
    if isinstance(payload, dict):
        results = payload.get("tool_results")
        if isinstance(results, list):
            return [_coerce_result(r) for r in results if isinstance(r, dict)]
    if isinstance(payload, list):
        return [_coerce_result(r) for r in payload if isinstance(r, dict)]
    return []


def _coerce_result(raw: dict) -> dict:
    call_id = raw.get("toolCallId") or raw.get("tool_call_id") or raw.get("id")
    content = raw.get("content")
    if content is None:
        content = raw.get("result")
    if not isinstance(content, str):
        content = json.dumps(content) if content is not None else ""
    return {"toolCallId": str(call_id) if call_id is not None else "", "content": content}
