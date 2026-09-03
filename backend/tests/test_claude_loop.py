"""Tests for the optional Claude Code agent loop (`backend/claude_loop/`).

No test spawns a real `claude` process: the SDK client is faked and the
subscription pre-flight / auth probe is monkeypatched. The billing tests assert
the **fail-closed** posture — a present `ANTHROPIC_API_KEY` must cause a
refuse-to-start, never an API-billed run.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    ToolUseBlock,
)
from fastapi import FastAPI

from pupa_backend.harnesses.claude import env as cl_env
from pupa_backend.harnesses.claude import frontend_tools, gate, registry
from pupa_backend.harnesses.claude.env import SubscriptionBillingUnavailable


# --------------------------------------------------------------------------- #
# Billing — subscription-only, fail-closed (the critical surface)
# --------------------------------------------------------------------------- #

def _clear_cred_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_build_sdk_env_excludes_forbidden_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-should-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")
    built = cl_env.build_sdk_env()
    for var in cl_env.FORBIDDEN_ENV_VARS:
        assert var not in built, f"{var} leaked into the SDK env"
    assert built.get("PATH") == "/usr/bin"


def test_assert_no_forbidden_env_raises_on_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-looking")
    with pytest.raises(SubscriptionBillingUnavailable) as exc:
        cl_env.assert_no_forbidden_env()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_assert_subscription_billing_refuses_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Critical negative test: a present API key → refuse-to-start, not an API run."""
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-looking")
    # Even if the probe would say 'subscription', the forbidden-var check fires first.
    monkeypatch.setattr(
        cl_env, "probe_auth_status",
        lambda env=None: {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"},
    )
    with pytest.raises(SubscriptionBillingUnavailable):
        cl_env.assert_subscription_billing()


@pytest.mark.parametrize(
    "probe",
    [
        {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"},
        {"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"},
        {"loggedIn": True, "authMethod": "third_party", "apiProvider": "bedrock"},
        {"loggedIn": True, "authMethod": "mystery", "apiProvider": "firstParty"},
    ],
)
def test_assert_subscription_billing_refuses_non_subscription(
    monkeypatch: pytest.MonkeyPatch, probe: dict
) -> None:
    _clear_cred_env(monkeypatch)
    monkeypatch.setattr(cl_env, "probe_auth_status", lambda env=None: probe)
    with pytest.raises(SubscriptionBillingUnavailable):
        cl_env.assert_subscription_billing()


@pytest.mark.parametrize("method", ["claude.ai", "oauth_token"])
def test_assert_subscription_billing_passes_for_subscription(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    _clear_cred_env(monkeypatch)
    monkeypatch.setattr(
        cl_env, "probe_auth_status",
        lambda env=None: {"loggedIn": True, "authMethod": method, "apiProvider": "firstParty"},
    )
    data = cl_env.assert_subscription_billing()
    assert data["authMethod"] == method


def test_assert_subscription_billing_refuses_api_billing_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_BILLING", "api")
    with pytest.raises(SubscriptionBillingUnavailable):
        cl_env.assert_subscription_billing()


# --------------------------------------------------------------------------- #
# Gate — can_use_tool policy
# --------------------------------------------------------------------------- #

async def _decide(callback, name, args=None):
    return await callback(name, dict(args or {}), _DummyCtx())


class _DummyCtx:
    pass


async def test_gate_allows_frontend_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "off")
    cb = gate.make_can_use_tool({})
    res = await _decide(cb, frontend_tools.qualified_name("renderChecklist"))
    assert isinstance(res, PermissionResultAllow)


async def test_gate_mutes_disabled_frontend_tool() -> None:
    cb = gate.make_can_use_tool({"disabled_tools": ["renderChecklist"]})
    res = await _decide(cb, frontend_tools.qualified_name("renderChecklist"))
    assert isinstance(res, PermissionResultDeny)


async def test_gate_native_scope_off_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "off")
    cb = gate.make_can_use_tool({})
    assert isinstance(await _decide(cb, "Read"), PermissionResultDeny)


async def test_gate_native_scope_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "read")
    cb = gate.make_can_use_tool({})
    assert isinstance(await _decide(cb, "Read"), PermissionResultAllow)
    assert isinstance(await _decide(cb, "Grep"), PermissionResultAllow)
    assert isinstance(await _decide(cb, "Edit"), PermissionResultDeny)
    assert isinstance(await _decide(cb, "Bash"), PermissionResultDeny)


async def test_gate_native_scope_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    cb = gate.make_can_use_tool({})
    assert isinstance(await _decide(cb, "Edit"), PermissionResultAllow)
    assert isinstance(await _decide(cb, "Bash"), PermissionResultAllow)
    assert isinstance(await _decide(cb, "Read"), PermissionResultAllow)


def test_auto_approved_native_tools_by_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "off")
    assert gate.auto_approved_native_tools() == []
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "read")
    assert "Read" in gate.auto_approved_native_tools() and "Edit" not in gate.auto_approved_native_tools()
    # Edit-class tools are NEVER pre-approved — they must route through can_use_tool
    # so the user can be asked for permission.
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    assert "Edit" not in gate.auto_approved_native_tools()
    assert "Read" in gate.auto_approved_native_tools()


# --------------------------------------------------------------------------- #
# #2 — user-approved command permission
# --------------------------------------------------------------------------- #

def test_interpret_approval() -> None:
    for yes in ("yes", "y", "OK", "ok do it", "sure", "approve", "go ahead", "Yes, proceed"):
        assert gate.interpret_approval(yes) is True, yes
    for no in ("no", "", "maybe later", "stop", "don't", "not now"):
        assert gate.interpret_approval(no) is False, no


def _decision(out: dict) -> str:
    return out["hookSpecificOutput"]["permissionDecision"]


async def test_hook_edit_tool_parks_for_user_then_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL", "1")
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_AUTO_APPROVE", raising=False)
    session = registry.LiveSession(thread_id="t-perm")
    session.current_run_id = "r1"
    session.run_open = True  # the endpoint has a run in flight (see `open_run`)
    hook = gate.make_pre_tool_use_hook({}, session)
    task = asyncio.ensure_future(hook({"tool_name": "Bash", "tool_input": {"command": "ls"}}, "tid", None))
    await asyncio.sleep(0.02)
    assert not task.done()  # parked awaiting the user's reply
    assert session.pending_decision is not None
    # The approval request + run_finished + INTERRUPT marker are queued.
    assert session.queue.qsize() >= 4

    session.pending_decision.set_result(True)
    out = await asyncio.wait_for(task, timeout=1.0)
    assert _decision(out) == "allow"


async def test_hook_edit_tool_denied_by_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL", "1")
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_AUTO_APPROVE", raising=False)
    session = registry.LiveSession(thread_id="t-perm2")
    hook = gate.make_pre_tool_use_hook({}, session)
    task = asyncio.ensure_future(hook({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, "tid", None))
    await asyncio.sleep(0.02)
    session.pending_decision.set_result(False)
    out = await asyncio.wait_for(task, timeout=1.0)
    assert _decision(out) == "deny"


async def test_hook_read_tool_no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    session = registry.LiveSession(thread_id="t-perm3")
    hook = gate.make_pre_tool_use_hook({}, session)
    out = await hook({"tool_name": "Read", "tool_input": {"file_path": "/x"}}, "tid", None)
    assert _decision(out) == "allow"
    assert session.pending_decision is None  # reads don't prompt


async def test_hook_denies_out_of_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "off")
    hook = gate.make_pre_tool_use_hook({}, None)
    out = await hook({"tool_name": "Bash", "tool_input": {"command": "ls"}}, "tid", None)
    assert _decision(out) == "deny"


async def test_hook_allows_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    hook = gate.make_pre_tool_use_hook({"disabled_tools": ["muted"]}, None)
    out = await hook({"tool_name": frontend_tools.qualified_name("renderChecklist"), "tool_input": {}}, "t", None)
    assert _decision(out) == "allow"
    out2 = await hook({"tool_name": frontend_tools.qualified_name("muted"), "tool_input": {}}, "t", None)
    assert _decision(out2) == "deny"


async def test_hook_ask_permission_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_ASK_PERMISSION", "0")
    session = registry.LiveSession(thread_id="t-perm4")
    hook = gate.make_pre_tool_use_hook({}, session)
    out = await hook({"tool_name": "Bash", "tool_input": {"command": "ls"}}, "tid", None)
    assert _decision(out) == "allow"  # auto-allow, no prompt
    assert session.pending_decision is None


def test_interpret_always() -> None:
    for yes in ("always", "yes, always", "auto", "approve all", "run freely", "yolo", "don't ask"):
        assert gate.interpret_always(yes) is True, yes
    for no in ("yes", "ok", "no", ""):
        assert gate.interpret_always(no) is False, no
    assert gate.interpret_approval("always") is True  # "always" also approves


@pytest.mark.parametrize("source", ["global", "state", "session"])
async def test_hook_run_freely_sources(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL", "1")  # would prompt, but overrides win
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_AUTO_APPROVE", raising=False)
    session = registry.LiveSession(thread_id=f"rf-{source}")
    state: dict = {}
    if source == "global":
        monkeypatch.setenv("PUPA_CLAUDE_LOOP_AUTO_APPROVE", "1")
    elif source == "state":
        state = {"claude_loop_auto_approve": True}
    else:
        session.auto_approve = True
    hook = gate.make_pre_tool_use_hook(state, session)
    out = await hook({"tool_name": "Bash", "tool_input": {"command": "ls"}}, "tid", None)
    assert _decision(out) == "allow"
    assert session.pending_decision is None  # no prompt


# --------------------------------------------------------------------------- #
# #1 — interactive built-ins disallowed; model asks in plain chat
# --------------------------------------------------------------------------- #

def test_loop_system_prompt_tells_model_to_ask_in_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses.claude.env import loop_system_prompt

    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "off")
    p = loop_system_prompt()
    assert "ask" in p.lower()
    assert "pop up" in p.lower() or "dialog" in p.lower()
    assert "host machine tools" not in p.lower()  # no host-tools section when off


def test_loop_system_prompt_adds_host_tools_when_native(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses.claude.env import loop_system_prompt

    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "full")
    p = loop_system_prompt()
    assert "host machine tools" in p.lower()
    assert "shell" in p.lower()


def test_full_scope_permits_all_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "full")
    assert gate.native_enabled() is True
    # Web reads auto-approved; arbitrary command tool permitted (asked).
    assert "WebFetch" in gate.auto_approved_native_tools()
    allow, needs_ask, _ = gate._resolve_static("Bash", set())
    assert allow and needs_ask
    allow_web, ask_web, _ = gate._resolve_static("WebSearch", set())
    assert allow_web and not ask_web
    allow_task, ask_task, _ = gate._resolve_static("Task", set())
    assert allow_task and ask_task  # unknown tool permitted in full, asked


def test_full_scope_alias_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "all")
    assert gate.resolve_native_scope() == "full"


def test_native_scope_defaults_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_NATIVE", raising=False)
    assert gate.resolve_native_scope() == "full"  # permissive default
    # Per-turn state override (app can switch plan=read / edit=full without restart).
    assert gate.resolve_native_scope({"claude_loop_native": "read"}) == "read"
    assert gate.resolve_native_scope({"claude_loop_native": "off"}) == "off"


def test_require_approval_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL", raising=False)
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_ASK_PERMISSION", raising=False)
    assert gate._require_approval() is False  # run-freely by default
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL", "1")
    assert gate._require_approval() is True


def test_interactive_builtins_disallowed() -> None:
    from pupa_backend.harnesses.claude.endpoint import _DISALLOWED_BUILTINS

    assert "AskUserQuestion" in _DISALLOWED_BUILTINS
    assert "ExitPlanMode" in _DISALLOWED_BUILTINS


# --------------------------------------------------------------------------- #
# Skills, setting_sources, config MCP bridge
# --------------------------------------------------------------------------- #

def test_loop_skills_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses.claude.env import loop_skills

    monkeypatch.delenv("PUPA_CLAUDE_LOOP_SKILLS", raising=False)
    assert loop_skills() is None
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_SKILLS", "off")
    assert loop_skills() is None
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_SKILLS", "all")
    assert loop_skills() == "all"
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_SKILLS", "foo, bar")
    assert loop_skills() == ["foo", "bar"]


def test_loop_setting_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses.claude.env import loop_setting_sources

    monkeypatch.delenv("PUPA_CLAUDE_LOOP_SKILLS", raising=False)
    assert loop_setting_sources() == []  # isolated when skills off
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_SKILLS", "all")
    assert loop_setting_sources() == ["user", "project"]  # loaded for skill discovery


async def test_build_config_mcp_bridges_shared_tools() -> None:
    """Config MCP servers are bridged in-process from the single shared connection.

    The bridge wraps each already-connected LangChain tool as an SDK MCP tool whose
    handler executes the tool in-process (against the shared session) — so every
    claude thread reuses one server instead of spawning its own subprocess copy.
    """
    from pupa_backend.harnesses.claude import config_mcp

    class FakeTool:
        name = "confluence_search"
        description = "search confluence"
        args_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def ainvoke(self, args: dict) -> str:
            self.calls.append(args)
            return "PAGE RESULTS"

    class FakeMcp:
        def __init__(self, tools: list) -> None:
            self.tools = tools

    ft = FakeTool()
    server, qualified = config_mcp.build_config_mcp(FakeMcp([ft]))
    assert server is not None
    assert qualified == {config_mcp.qualified_name("confluence_search")}

    # The handler runs the shared tool in-process and wraps the result as MCP content.
    out = await config_mcp._make_handler(ft)({"query": "onboarding"})
    assert ft.calls == [{"query": "onboarding"}]
    assert out == {"content": [{"type": "text", "text": "PAGE RESULTS"}]}

    # Nothing configured/connected → no server (frontend tools still run alone).
    assert config_mcp.build_config_mcp(None) == (None, set())
    assert config_mcp.build_config_mcp(FakeMcp([])) == (None, set())


async def test_gate_allows_external_mcp_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "off")
    allow, needs_ask, _ = gate._resolve_static("mcp__atlassian__search", set())
    assert allow and not needs_ask  # operator-configured MCP tool, no prompt


def test_skill_tools_auto_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "full")
    allow, needs_ask, _ = gate._resolve_static("Skill", set())
    assert allow and not needs_ask


# --------------------------------------------------------------------------- #
# Frontend tools → in-process MCP + registry correlation
# --------------------------------------------------------------------------- #

async def test_build_frontend_mcp_names_and_parking() -> None:
    session = registry.LiveSession(thread_id="t-fe")
    descriptors = [
        {"name": "renderChecklist", "description": "render", "parameters": {"type": "object"}},
        {"name": "addComponent", "description": "add", "parameters": {"type": "object"}},
    ]
    server, qualified = frontend_tools.build_frontend_mcp(descriptors, session)
    assert server is not None
    assert qualified == {
        frontend_tools.qualified_name("renderChecklist"),
        frontend_tools.qualified_name("addComponent"),
    }


async def test_registry_claim_after_resolve() -> None:
    """Handler runs AFTER resume delivers the result (the live-SDK ordering)."""
    session = registry.LiveSession(thread_id="t-corr")
    # Pump records the call; resume delivers the result before the handler runs.
    await session.register_pending("call-1", "renderChecklist", {"items": [1, 2]})
    await session.resolve_results([{"toolCallId": "call-1", "content": "done"}])

    # The handler (claim_call) now consumes the result slot by (name, args).
    result = await asyncio.wait_for(
        session.claim_call("renderChecklist", {"items": [1, 2]}, timeout=1.0), timeout=1.0
    )
    assert result == {"content": [{"type": "text", "text": "done"}]}


async def test_registry_claim_waits_for_late_result() -> None:
    """Handler runs BEFORE register/resolve: claim must wait, not fail."""
    session = registry.LiveSession(thread_id="t-late")
    claim_task = asyncio.ensure_future(session.claim_call("foo", {"a": 1}, timeout=2.0))
    await asyncio.sleep(0.05)
    assert not claim_task.done()
    await session.register_pending("c9", "foo", {"a": 1})
    await asyncio.sleep(0.01)
    assert not claim_task.done()  # result not delivered yet
    await session.resolve_results([{"toolCallId": "c9", "content": "late"}])
    result = await asyncio.wait_for(claim_task, timeout=1.0)
    assert result == {"content": [{"type": "text", "text": "late"}]}


async def test_registry_resolve_synthesises_missing_results() -> None:
    session = registry.LiveSession(thread_id="t-missing")
    await session.register_pending("c-miss", "foo", {})
    # Resume with no matching result → the slot is filled with an error so the SDK
    # tool handler returns instead of hanging.
    await session.resolve_results([])
    result = await asyncio.wait_for(session.claim_call("foo", {}, timeout=1.0), timeout=1.0)
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"ok": False, "error": "missing_tool_result"}


# --------------------------------------------------------------------------- #
# Endpoint round-trip with a faked SDK client
# --------------------------------------------------------------------------- #

class _FakeSDKClient:
    """Stand-in for `ClaudeSDKClient` that drives one frontend-tool round-trip.

    `receive_messages()` yields an assistant message calling a frontend tool, then
    blocks on the registered pending future(s) (as the real SDK blocks waiting for
    the tool result) until the resume POST resolves them, then yields the terminal
    `ResultMessage`.
    """

    def __init__(self, options=None, transport=None):
        self.options = options
        self.thread_id = "thread-rt"

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self._prompt = prompt

    async def disconnect(self):
        return None

    async def receive_messages(self):
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="call-rt",
                    name=frontend_tools.qualified_name("renderChecklist"),
                    input={"items": ["a"]},
                )
            ],
            model="fake",
            message_id="m-rt",
        )
        # Drive the in-process tool handler exactly as the real SDK would: once the
        # pump has recorded the frontend call, invoke claim_call and block on its
        # result (delivered by the resume POST), then finish the turn.
        sess = registry.get(self.thread_id)
        for _ in range(100000):
            if sess and sess.pending:
                break
            await asyncio.sleep(0)
        await sess.claim_call("renderChecklist", {"items": ["a"]})
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-sess-rt",
        )


def _sse_events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


async def test_endpoint_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cred_env(monkeypatch)
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _FakeSDKClient)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # --- First POST: expect a batched on_interrupt then run_finished ---
        body = {
            "thread_id": "thread-rt",
            "run_id": "run-1",
            "messages": [{"id": "u1", "role": "user", "content": "make a checklist"}],
            "tools": [{"name": "renderChecklist", "description": "r", "parameters": {"type": "object"}}],
            "state": {},
            "context": [],
            "forwardedProps": {},
        }
        r1 = await client.post("/", json=body)
        evs1 = _sse_events(r1.text)
        types1 = [e["type"] for e in evs1]
        assert "RUN_STARTED" in types1
        assert "CUSTOM" in types1
        custom = next(e for e in evs1 if e["type"] == "CUSTOM")
        assert custom["name"] == "on_interrupt"
        calls = custom["value"]["frontend_tool_calls"]
        assert calls == [{"id": "call-rt", "name": "renderChecklist", "args": {"items": ["a"]}}]
        assert "RUN_FINISHED" in types1

        # Session stays parked between requests.
        assert registry.get("thread-rt") is not None

        # --- Resume POST: deliver the on-device result; expect the turn to finish ---
        resume_body = {
            "thread_id": "thread-rt",
            "run_id": "run-2",
            "messages": [],
            "tools": [],
            "state": {},
            "context": [],
            "forwardedProps": {
                "command": {"resume": {"tool_results": [{"toolCallId": "call-rt", "content": "ok"}]}}
            },
        }
        r2 = await client.post("/", json=resume_body)
        types2 = [e["type"] for e in _sse_events(r2.text)]
        assert "RUN_STARTED" in types2
        assert "RUN_FINISHED" in types2

    # Session removed after the run finished.
    assert registry.get("thread-rt") is None


async def test_endpoint_permission_reply_resolves_parked_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new-turn POST while a permission request is parked resolves it."""
    _clear_cred_env(monkeypatch)
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {})
    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    # Seed a parked session awaiting a permission decision, with a tiny "pump"
    # that finishes the turn once the decision is resolved.
    session = registry.create("perm-thread")
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    session.pending_decision = fut
    session.current_run_id = "r0"

    async def _fake_pump():
        allow = await fut
        session.emit(cl_endpoint.events.run_finished("perm-thread", session.current_run_id or ""))
        session.mark_finish()
        session._allow = allow  # type: ignore[attr-defined]

    session.pump_task = asyncio.ensure_future(_fake_pump())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "thread_id": "perm-thread",
            "run_id": "r1",
            "messages": [{"id": "u", "role": "user", "content": "yes"}],
            "tools": [],
            "state": {},
            "context": [],
            "forwardedProps": {},
        }
        r = await client.post("/", json=body)
        types = [e["type"] for e in _sse_events(r.text)]
        assert "RUN_STARTED" in types and "RUN_FINISHED" in types
    assert getattr(session, "_allow", None) is True


async def test_endpoint_permission_always_sets_auto_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replying 'always' flips the thread to run-freely."""
    _clear_cred_env(monkeypatch)
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {})
    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    session = registry.create("always-thread")
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    session.pending_decision = fut
    session.current_run_id = "r0"

    async def _fake_pump():
        await fut
        session.emit(cl_endpoint.events.run_finished("always-thread", session.current_run_id or ""))
        session.mark_finish()

    session.pump_task = asyncio.ensure_future(_fake_pump())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "thread_id": "always-thread", "run_id": "r1",
            "messages": [{"id": "u", "role": "user", "content": "always"}],
            "tools": [], "state": {}, "context": [], "forwardedProps": {},
        }
        await client.post("/", json=body)
    assert session.auto_approve is True
    assert fut.result() is True


async def test_endpoint_resume_without_session_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cred_env(monkeypatch)
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _FakeSDKClient)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "thread_id": "ghost-thread",
            "run_id": "r",
            "messages": [],
            "tools": [],
            "state": {},
            "context": [],
            "forwardedProps": {"command": {"resume": {"tool_results": []}}},
        }
        r = await client.post("/", json=body)
        types = [e["type"] for e in _sse_events(r.text)]
        assert types == ["RUN_ERROR"]


# --------------------------------------------------------------------------- #
# Multimodal input: images on user messages reach the model
# --------------------------------------------------------------------------- #

class _CapturingSDKClient:
    """Captures whatever `query()` receives, then finishes the turn immediately."""

    captured: object = None

    def __init__(self, options=None, transport=None):
        self.options = options

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        if isinstance(prompt, str):
            _CapturingSDKClient.captured = prompt
        else:
            msgs = []
            async for m in prompt:
                msgs.append(m)
            _CapturingSDKClient.captured = msgs

    async def disconnect(self):
        return None

    async def receive_messages(self):
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-img",
        )


async def test_endpoint_forwards_image_parts_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user message with an image part is streamed to the SDK as an Anthropic
    image block, not dropped down to text-only."""
    _clear_cred_env(monkeypatch)
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    _CapturingSDKClient.captured = None
    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _CapturingSDKClient)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "thread_id": "img-thread",
            "run_id": "r1",
            "messages": [{
                "id": "u1",
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image", "source": {
                        "type": "data", "value": "BASE64DATA", "mime_type": "image/png"}},
                ],
            }],
            "tools": [],
            "state": {},
            "context": [],
            "forwardedProps": {},
        }
        r = await client.post("/", json=body)
        assert "RUN_FINISHED" in [e["type"] for e in _sse_events(r.text)]

    captured = _CapturingSDKClient.captured
    assert isinstance(captured, list), "image turn must stream a structured message, not a string"
    content = captured[0]["message"]["content"]
    text_blocks = [b for b in content if b.get("type") == "text"]
    assert text_blocks and "what is this" in text_blocks[0]["text"]
    images = [b for b in content if b.get("type") == "image"]
    assert images == [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "BASE64DATA"},
    }]

    await registry.remove("img-thread")


async def test_endpoint_text_only_stays_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: text-only turns keep the plain-string query path."""
    _clear_cred_env(monkeypatch)
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    _CapturingSDKClient.captured = None
    monkeypatch.setattr(cl_endpoint, "assert_subscription_billing", lambda: {})
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _CapturingSDKClient)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "thread_id": "txt-thread", "run_id": "r1",
            "messages": [{"id": "u1", "role": "user", "content": "hello there"}],
            "tools": [], "state": {}, "context": [], "forwardedProps": {},
        }
        await client.post("/", json=body)

    assert _CapturingSDKClient.captured == "user: hello there"
    await registry.remove("txt-thread")


def test_image_block_url_source() -> None:
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint

    part = {"type": "image", "source": {"type": "url", "value": "https://x/y.jpg"}}
    assert cl_endpoint._image_block(part) == {
        "type": "image", "source": {"type": "url", "url": "https://x/y.jpg"},
    }


# --------------------------------------------------------------------------- #
# Dispatch: the harness registry selects which harnesses mount
# --------------------------------------------------------------------------- #

def test_registry_default_is_langgraph_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses import build_registry

    monkeypatch.delenv("PUPA_HARNESSES", raising=False)
    reg = build_registry()
    assert reg.ids() == ["deepagents"]
    assert reg.default().id == "deepagents"


def test_registry_mounts_both_harnesses(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from pupa_backend.harnesses import build_registry

    monkeypatch.setenv(
        "PUPA_HARNESSES",
        json.dumps(
            {
                "deepagents": {"enabled": True, "default": True},
                "claude_code": {"enabled": True},
            }
        ),
    )
    reg = build_registry()
    assert set(reg.ids()) == {"deepagents", "claude_code"}
    assert reg.default().id == "deepagents"
