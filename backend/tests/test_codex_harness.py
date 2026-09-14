"""Unit tests for the Codex App Server harness."""

from __future__ import annotations

import asyncio
import os

import pytest
from ag_ui.core import EventType

from pupa_backend.agui.input import image_inputs
from pupa_backend.harnesses.codex import env
from pupa_backend.harnesses.codex.models import ModelCatalog
from pupa_backend.harnesses.codex import registry
from pupa_backend.harnesses.codex.registry import BOUNDARY, LiveSession
from pupa_backend.harnesses.codex.tools import (
    FRONTEND_NAMESPACE,
    MCP_NAMESPACE,
    build_tool_surface,
)


def _catalog() -> ModelCatalog:
    return ModelCatalog(
        [
            {
                "id": "codex-a",
                "model": "codex-a",
                "displayName": "Codex A",
                "isDefault": True,
                "hidden": False,
                "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "low", "description": ""},
                    {"reasoningEffort": "medium", "description": ""},
                ],
            },
            {
                "id": "codex-hidden",
                "model": "codex-hidden",
                "displayName": "Hidden",
                "isDefault": False,
                "hidden": True,
                "defaultReasoningEffort": "high",
                "supportedReasoningEfforts": [],
            },
        ]
    )


def test_catalog_exposes_models_and_reasoning() -> None:
    catalog = ModelCatalog.from_response({"data": _catalog().data})
    assert catalog.menu() == [
        {"provider": "codex", "modelId": "codex-a", "label": "Codex A"}
    ]
    assert catalog.thinking_menu() == [
        {"level": "low", "label": "Low"},
        {"level": "medium", "label": "Medium"},
    ]
    assert catalog.default_model() == "codex-a"


def test_catalog_resolves_request_and_falls_back() -> None:
    class Input:
        forwarded_props = {"llm": {"model": "codex-a", "thinking": "low"}}

    catalog = _catalog()
    assert catalog.resolve_model(Input()) == "codex-a"
    assert catalog.resolve_effort(Input(), "codex-a") == "low"

    Input.forwarded_props = {"llm": {"model": "elsewhere", "thinking": "ultra"}}
    assert catalog.resolve_model(Input()) == "codex-a"
    assert catalog.resolve_effort(Input(), "codex-a") == "medium"


def test_child_env_excludes_api_billing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("CODEX_API_KEY", "secret")
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("PUPA_CODEX_CONFIG_DIR", "/tmp/codex-home")
    child = env.child_env()
    assert child["PATH"] == "/bin"
    assert child["CODEX_HOME"] == "/tmp/codex-home"
    assert "OPENAI_API_KEY" not in child
    assert "CODEX_API_KEY" not in child


def test_sandbox_mapping_and_state_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_CODEX_LOOP_NATIVE", "read")
    assert env.native_scope() == "read"
    assert env.native_scope({"codex_loop_native": "full"}) == "full"
    assert env.turn_sandbox("read", "/work") == {
        "type": "readOnly",
        "networkAccess": False,
    }
    assert env.turn_sandbox("workspace", "/work") == {
        "type": "workspaceWrite",
        "writableRoots": ["/work"],
        "networkAccess": False,
    }
    assert env.turn_sandbox("full", "/work") == {"type": "dangerFullAccess"}


def test_inline_image_becomes_data_url() -> None:
    content = [
        {
            "type": "image",
            "source": {"type": "base64", "mime_type": "image/png", "value": "abc"},
        }
    ]
    assert image_inputs(content) == [{"type": "image", "url": "data:image/png;base64,abc"}]


class _McpTool:
    name = "lookup"
    description = "Look something up"
    args_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    async def ainvoke(self, args):
        return {"answer": args["q"]}


class _Mcp:
    tools = [_McpTool()]


def test_tool_surface_namespaces_frontend_and_mcp() -> None:
    surface = build_tool_surface(
        [{"name": "open_card", "description": "Open", "parameters": {"type": "object"}}],
        {"disabled_tools": []},
        _Mcp(),
    )
    assert set(surface.frontend) == {"open_card"}
    assert set(surface.mcp) == {"lookup"}
    assert {row["name"] for row in surface.dynamic_tools} == {
        FRONTEND_NAMESPACE,
        MCP_NAMESPACE,
    }


def test_invalid_and_duplicate_frontend_tool_names_fail() -> None:
    with pytest.raises(ValueError, match="not valid"):
        build_tool_surface([{"name": "bad name"}], {}, None)
    with pytest.raises(ValueError, match="duplicate"):
        build_tool_surface([{"name": "same"}, {"name": "same"}], {}, None)


async def test_changed_tool_surface_starts_fresh_codex_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.calls = []
            self.closed = False
            instances.append(self)

        async def connect(self):
            return {}

        async def request(self, method, params, **kwargs):
            self.calls.append((method, params))
            if method == "account/read":
                return {"account": {"type": "chatgpt"}}
            if method == "thread/start":
                return {"thread": {"id": f"codex-{len(instances)}", "modelProvider": "openai"}}
            raise AssertionError(f"unexpected request: {method}")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(registry, "AppServerClient", Client)
    first = build_tool_surface([{"name": "first"}], {}, None)
    widened = build_tool_surface([{"name": "first"}, {"name": "second"}], {}, None)
    session = LiveSession("surface-thread", None)
    try:
        await session.connect(first)
        assert registry.remembered_surface("surface-thread") == first.fingerprint
        assert await session.reload_surface(first) is False
        assert len(instances) == 1
        assert await session.reload_surface(widened) is True
        assert len(instances) == 2
        assert instances[0].closed is True
        assert [method for method, _params in instances[1].calls].count("thread/start") == 1
        assert all(method != "thread/resume" for method, _params in instances[1].calls)
        assert registry.remembered_surface("surface-thread") == widened.fingerprint
    finally:
        await session.dispose(notify=False)
        registry.forget_thread_id("surface-thread")


async def test_unusable_remembered_thread_falls_back_to_fresh_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            return {}

        async def request(self, method, params, **kwargs):
            calls.append((method, params))
            if method == "account/read":
                return {"account": {"type": "chatgpt"}}
            if method == "thread/resume":
                raise registry.AppServerError("remembered thread no longer exists")
            if method == "thread/start":
                return {"thread": {"id": "fresh-codex-thread", "modelProvider": "openai"}}
            raise AssertionError(f"unexpected request: {method}")

        async def close(self):
            pass

    monkeypatch.setattr(registry, "AppServerClient", Client)
    surface = build_tool_surface([{"name": "pick"}], {}, None)
    registry.remember_thread_id("pupa-thread", "stale-codex-thread", surface.fingerprint)
    session = LiveSession("pupa-thread", None)
    try:
        resumed = await session.connect(surface, resume_id="stale-codex-thread")

        assert resumed is False
        assert [method for method, _params in calls] == [
            "account/read",
            "thread/resume",
            "thread/start",
        ]
        assert session.codex_thread_id == "fresh-codex-thread"
        assert registry.remembered_thread_id("pupa-thread") == "fresh-codex-thread"
    finally:
        await session.dispose(notify=False)
        registry.forget_thread_id("pupa-thread")


async def test_frontend_tool_interrupt_and_resume() -> None:
    session = LiveSession("thread", None)
    session.surface = build_tool_surface([{"name": "pick"}], {}, None)
    session.open_run("run-1")
    request = asyncio.create_task(
        session.on_request(
            "item/tool/call",
            {
                "namespace": FRONTEND_NAMESPACE,
                "tool": "pick",
                "callId": "call-1",
                "arguments": {"value": 1},
            },
        )
    )
    await asyncio.sleep(0.03)
    queued = []
    while True:
        item = await session.queue.get()
        if item is BOUNDARY:
            break
        queued.append(item)
    assert any(getattr(item, "type", None) == EventType.TOOL_CALL_START for item in queued)
    interrupt = next(item for item in queued if getattr(item, "name", None) == "on_interrupt")
    assert interrupt.value["frontend_tool_calls"] == [
        {"id": "call-1", "name": "pick", "args": {"value": 1}}
    ]

    await session.resolve_tool_results([{"toolCallId": "call-1", "content": "chosen"}])
    assert await request == {
        "contentItems": [{"type": "inputText", "text": "chosen"}],
        "success": True,
    }


async def test_approval_reply_and_auto_approve() -> None:
    session = LiveSession("thread", None)
    session.scope = "full"
    session.open_run("run-1")
    request = asyncio.create_task(
        session.on_request(
            "item/commandExecution/requestApproval",
            {"command": "git status", "itemId": "item", "threadId": "c", "turnId": "t"},
        )
    )
    await asyncio.sleep(0)
    assert session.pending_approval is not None
    await session.resolve_approval(True, always=True)
    assert await request == {"decision": "acceptForSession"}

    second = await session.on_request(
        "item/commandExecution/requestApproval",
        {"command": "git diff", "itemId": "item2", "threadId": "c", "turnId": "t"},
    )
    assert second == {"decision": "acceptForSession"}


async def test_permission_escalation_never_crosses_scope_ceiling() -> None:
    session = LiveSession("thread", None)
    session.scope = "workspace"
    session.auto_approve_commands = True
    response = await session.on_request(
        "item/permissions/requestApproval",
        {
            "permissions": {"network": {"enabled": True}},
            "itemId": "item",
            "threadId": "c",
            "turnId": "t",
        },
    )
    assert response == {"permissions": {}, "scope": "turn"}


@pytest.mark.parametrize(
    "scope,method",
    [
        ("read", "item/commandExecution/requestApproval"),
        ("workspace", "item/commandExecution/requestApproval"),
        ("workspace", "item/fileChange/requestApproval"),
    ],
)
async def test_command_and_file_approvals_do_not_widen_sandbox(
    scope: str,
    method: str,
) -> None:
    session = LiveSession("thread", None, scope=scope, auto_approve_commands=True)
    response = await session.on_request(method, {"command": "touch /outside"})
    assert response == {"decision": "decline"}


async def test_second_stream_attach_gets_every_event() -> None:
    session = LiveSession("thread", None)
    first: list = []
    second: list = []

    async def drain(output: list) -> None:
        async for event in registry.attach(session):
            output.append(event)

    first_task = asyncio.create_task(drain(first))
    await asyncio.sleep(0)
    second_task = asyncio.create_task(drain(second))
    await asyncio.sleep(0)
    session.emit("one")
    session.emit("two")
    session.queue.put_nowait(BOUNDARY)
    await asyncio.wait_for(second_task, timeout=2)
    await asyncio.wait_for(first_task, timeout=2)
    assert first == []
    assert second == ["one", "two"]


async def test_surface_continuation_survives_fast_old_turn_completion() -> None:
    class Client:
        async def request(self, method, params, **kwargs):
            assert method == "turn/start"
            return {"turn": {"id": "old-turn"}}

    session = LiveSession("thread", None)
    session.client = Client()
    session.codex_thread_id = "codex-thread"
    await session.start_turn(
        text="latest request",
        images=[],
        model=None,
        effort=None,
        state=None,
        transcript="user: earlier context\n\nuser: latest request",
    )
    session._completed_frontend = ["- unlock_probe({}) => unlocked"]
    session.arm_surface_continuation()

    await session.on_notification(
        "turn/completed",
        {"turn": {"id": "old-turn", "status": "completed"}},
    )
    assert session.turn_active is False
    assert session.queue.empty()

    started = {}

    async def reload_surface(_surface):
        return True

    async def start_turn(**kwargs):
        started.update(kwargs)

    session.reload_surface = reload_surface
    session.start_turn = start_turn
    await session.continue_with_surface(
        build_tool_surface([{"name": "finish_probe"}], {}, None),
        model="codex-a",
        effort="medium",
        state={"codex_loop_native": "read"},
    )
    assert session._continuing is False
    assert started["preserve_request"] is True
    assert "earlier context" in started["text"]
    assert "latest request" in started["text"]
    assert "unlock_probe" in started["text"]


async def test_text_notifications_finish_an_agui_run() -> None:
    session = LiveSession("thread", None)
    session.open_run("run")
    await session.on_notification(
        "item/agentMessage/delta",
        {"itemId": "message", "delta": "hello", "threadId": "c", "turnId": "t"},
    )
    await session.on_notification(
        "item/completed",
        {"item": {"id": "message", "type": "agentMessage", "text": "hello"}},
    )
    await session.on_notification(
        "turn/completed",
        {"turn": {"id": "t", "status": "completed"}},
    )
    found = []
    while True:
        item = await session.queue.get()
        if item is BOUNDARY:
            break
        found.append(item)
    assert [item.type for item in found] == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.RUN_FINISHED,
    ]
