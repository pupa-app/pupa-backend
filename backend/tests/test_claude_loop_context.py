"""The Claude-loop harness must forward `RunAgentInput.context` into the turn.

The frontend pushes ambient context (live canvas state, memories snapshot, the
MyApp system prompt / AGENTS.md) every turn as `context: [{description, value}]`.
Unlike the LangGraph harness, the loop builds its own prompt — so it has to
render those entries into the system prompt itself, else the model never sees
the app's instructions or canvas state (issue: context dropped). Placement is
the system-prompt end (refreshes in place each turn, stays cacheable), not a
per-turn user message (which would accumulate in the transcript).
"""

from __future__ import annotations

import httpx
import pytest
from claude_agent_sdk import ResultMessage
from fastapi import FastAPI

from pupa_backend.harnesses.claude import endpoint as cl_endpoint
from pupa_backend.harnesses.claude import env as cl_env
from pupa_backend.harnesses.claude import registry


class _FinishingClient:
    """Fake `ClaudeSDKClient` that records queries and finishes immediately."""

    instances: list["_FinishingClient"] = []

    def __init__(self, options=None, transport=None):
        self.options = options
        self.queries: list = []
        _FinishingClient.instances.append(self)

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self.queries.append(prompt)

    async def interrupt(self):
        return None

    async def disconnect(self):
        return None

    async def receive_messages(self):
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1,
            is_error=False, num_turns=1, session_id="sdk-sess-ctx",
        )


@pytest.fixture
def loop_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _FinishingClient)
    _FinishingClient.instances = []
    registry._REGISTRY.clear()
    registry._SESSION_IDS.clear()
    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)
    return app


# --------------------------------------------------------------------------- #
# Pure helper
# --------------------------------------------------------------------------- #

def test_render_context_joins_description_and_value() -> None:
    from ag_ui.core import Context

    out = cl_endpoint._render_context([
        Context(description="Live canvas state — schema.", value='{"components":[]}'),
        Context(description="MyApp instructions (AGENTS.md).", value='{"typeId":"tracker"}'),
    ])
    assert "Live canvas state — schema." in out
    assert '{"components":[]}' in out
    assert "MyApp instructions (AGENTS.md)." in out
    assert '{"typeId":"tracker"}' in out
    # description precedes its value
    assert out.index("Live canvas state — schema.") < out.index('{"components":[]}')


def test_render_context_empty_is_blank() -> None:
    assert cl_endpoint._render_context([]) == ""
    assert cl_endpoint._render_context(None) == ""


# --------------------------------------------------------------------------- #
# End-to-end: context reaches the model via the system prompt (not the message,
# so it refreshes in place each turn instead of accumulating in the transcript)
# --------------------------------------------------------------------------- #

async def test_context_appended_to_system_prompt(loop_app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json={
            "thread_id": "thread-ctx", "run_id": "run-1",
            "messages": [{"id": "u1", "role": "user", "content": "add a books row"}],
            "tools": [], "state": {},
            "context": [
                {"description": "Live canvas state — thin enum.", "value": '{"components":[{"id":"tracker-1"}]}'},
                {"description": "MyApp instructions (pupa/AGENTS.md).", "value": '{"typeId":"tracker"}'},
            ],
            "forwardedProps": {},
        })

    assert _FinishingClient.instances, "no client was constructed"
    system_prompt = _FinishingClient.instances[0].options.system_prompt
    # Ambient context rides the system prompt…
    assert "Live canvas state — thin enum." in system_prompt
    assert '{"components":[{"id":"tracker-1"}]}' in system_prompt
    assert "MyApp instructions (pupa/AGENTS.md)." in system_prompt
    # …appended at the end, after the base loop prompt (stable prefix stays cacheable).
    assert system_prompt.index("Ambient context") > 0
    # The user's message is NOT bloated with the context (it stays in the transcript).
    sent = "\n".join(str(q) for c in _FinishingClient.instances for q in c.queries)
    assert "add a books row" in sent
    assert "Live canvas state — thin enum." not in sent
