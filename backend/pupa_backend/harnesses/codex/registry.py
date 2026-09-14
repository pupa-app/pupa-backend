"""Live Codex App Server sessions keyed by Pupa thread id."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ag_ui.core import EventType
from ag_ui.core.events import TextMessageContentEvent, TextMessageEndEvent, TextMessageStartEvent

from pupa_backend.agui import events
from pupa_backend.agui.stream import Attachment, single_consumer_events

from .env import (
    CodexSubscriptionUnavailable,
    auto_approve,
    child_env,
    codex_binary,
    codex_workspace,
    developer_instructions,
    native_scope,
    thread_sandbox,
    turn_sandbox,
)
from .tools import FRONTEND_NAMESPACE, MCP_NAMESPACE, ToolSurface, invoke_mcp
from .transport import AppServerClient, AppServerError

logger = logging.getLogger("uvicorn.error")

BOUNDARY = object()
ERROR = object()
_DEFAULT_IDLE_TIMEOUT = 900.0


def _idle_timeout() -> float:
    raw = os.getenv("PUPA_CODEX_IDLE_TIMEOUT")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("codex harness: invalid PUPA_CODEX_IDLE_TIMEOUT=%r", raw)
    return _DEFAULT_IDLE_TIMEOUT


def _thread_from(result: dict[str, Any]) -> dict[str, Any]:
    thread = result.get("thread") or {}
    return thread if isinstance(thread, dict) else {}


async def probe() -> list[dict[str, Any]]:
    """Validate CLI/authentication and return the account's visible model catalog."""
    client = AppServerClient(
        codex_binary(), env=child_env(), cwd=codex_workspace()
    )
    try:
        await client.connect()
        account_result = await client.request(
            "account/read", {"refreshToken": False}, timeout=15.0
        )
        account = account_result.get("account") or {}
        if not isinstance(account, dict) or account.get("type") != "chatgpt":
            account_type = account.get("type") if isinstance(account, dict) else None
            raise CodexSubscriptionUnavailable(
                "Codex harness requires a ChatGPT subscription login; "
                f"found {account_type or 'no login'}. Run `codex login`."
            )
        response = await client.request(
            "model/list", {"includeHidden": False, "limit": 100}, timeout=30.0
        )
        rows = response.get("data") or []
        if not isinstance(rows, list) or not rows:
            raise CodexSubscriptionUnavailable(
                "Codex login succeeded but the account returned no available models."
            )
        visible = [row for row in rows if isinstance(row, dict)]
        default = next((row for row in visible if row.get("isDefault")), visible[0])
        capability = await client.request(
            "thread/start",
            {
                "cwd": codex_workspace(),
                "model": default.get("model") or default.get("id"),
                "modelProvider": "openai",
                "approvalPolicy": "never",
                "approvalsReviewer": "user",
                "sandbox": "read-only",
                "developerInstructions": "Pupa Codex App Server capability probe.",
                "ephemeral": True,
                "serviceName": "pupa-backend",
                "dynamicTools": [
                    {
                        "type": "function",
                        "name": "pupa_capability_probe",
                        "description": "Pupa startup capability probe; never call.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ],
            },
            timeout=60.0,
        )
        provider = _thread_from(capability).get("modelProvider") or capability.get("modelProvider")
        if provider != "openai":
            raise CodexSubscriptionUnavailable(
                f"Codex App Server selected unsupported provider {provider!r}."
            )
        return visible
    except AppServerError as exc:
        raise CodexSubscriptionUnavailable(str(exc)) from exc
    finally:
        await client.close()


@dataclass
class PendingTool:
    call_id: str
    name: str
    arguments: dict[str, Any]
    future: asyncio.Future


@dataclass
class LiveSession:
    thread_id: str
    mcp: Any
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    attachment: Attachment | None = None
    pushback: list[Any] = field(default_factory=list)
    client: AppServerClient | None = None
    codex_thread_id: str | None = None
    surface: ToolSurface | None = None
    current_run_id: str | None = None
    current_turn_id: str | None = None
    turn_active: bool = False
    auto_approve_commands: bool = False
    scope: str = "workspace"
    last_activity: float = field(default_factory=time.monotonic)
    disposed: bool = False
    open_text: set[str] = field(default_factory=set)
    seen_items: set[str] = field(default_factory=set)
    pending_tools: dict[str, PendingTool] = field(default_factory=dict)
    active_batch: set[str] = field(default_factory=set)
    pending_approval: asyncio.Future | None = None
    approval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _batch_calls: list[dict[str, Any]] = field(default_factory=list)
    _batch_task: asyncio.Task | None = None
    _continuing: bool = False
    _turn_done: asyncio.Event = field(default_factory=asyncio.Event)
    _request_text: str = ""
    _request_transcript: str = ""
    _request_images: list[dict[str, Any]] = field(default_factory=list)
    _completed_frontend: list[str] = field(default_factory=list)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def emit(self, event: Any) -> None:
        if not self.disposed:
            self.queue.put_nowait(event)
            self.touch()

    def open_run(self, run_id: str) -> None:
        self.current_run_id = run_id
        self.emit(events.run_started(self.thread_id, run_id))

    def _close_text(self) -> None:
        for message_id in list(self.open_text):
            self.emit(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id))
        self.open_text.clear()

    def end_http_run(self, *, error: str | None = None) -> None:
        self._close_text()
        run_id = self.current_run_id
        if error:
            self.emit(events.run_error(error))
        elif run_id:
            self.emit(events.run_finished(self.thread_id, run_id))
        self.current_run_id = None
        self.queue.put_nowait(ERROR if error else BOUNDARY)

    async def connect(self, surface: ToolSurface, resume_id: str | None = None) -> bool:
        self.surface = surface
        self.client = AppServerClient(
            codex_binary(),
            env=child_env(),
            cwd=codex_workspace(),
            on_notification=self.on_notification,
            on_request=self.on_request,
        )
        await self.client.connect()
        account_result = await self.client.request(
            "account/read", {"refreshToken": False}, timeout=15.0
        )
        account = account_result.get("account") or {}
        if not isinstance(account, dict) or account.get("type") != "chatgpt":
            raise CodexSubscriptionUnavailable(
                "Codex harness requires a ChatGPT subscription login. Run `codex login`."
            )
        params: dict[str, Any] = {
            "cwd": codex_workspace(),
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "sandbox": thread_sandbox(self.scope),
            "developerInstructions": developer_instructions(),
            "modelProvider": "openai",
            "serviceName": "pupa-backend",
            # Experimental App Server field, enabled in initialize capabilities.
            "dynamicTools": surface.dynamic_tools,
        }
        resumed = False
        if resume_id:
            # Dynamic tools are fixed at thread creation. Resume is safe only
            # when the remembered surface fingerprint matches (the endpoint
            # enforces that before supplying `resume_id`).
            resume_params = dict(params)
            resume_params.pop("serviceName", None)
            resume_params.pop("dynamicTools", None)
            resume_params["threadId"] = resume_id
            try:
                result = await self.client.request(
                    "thread/resume", resume_params, timeout=60.0
                )
                resumed = True
            except AppServerError as exc:
                logger.warning(
                    "codex harness: remembered thread %s could not resume; "
                    "starting fresh: %s",
                    resume_id,
                    exc,
                )
                forget_thread_id(self.thread_id)
                result = await self.client.request("thread/start", params, timeout=60.0)
        else:
            result = await self.client.request("thread/start", params, timeout=60.0)
        thread = _thread_from(result)
        provider = thread.get("modelProvider") or result.get("modelProvider")
        if provider and provider != "openai":
            raise CodexSubscriptionUnavailable(
                f"Codex harness requires the OpenAI provider, found {provider!r}."
            )
        self.codex_thread_id = str(thread.get("id") or resume_id or "") or None
        if self.codex_thread_id:
            remember_thread_id(
                self.thread_id, self.codex_thread_id, surface.fingerprint
            )
        return resumed

    async def reload_surface(self, surface: ToolSurface) -> bool:
        """Install a changed surface on a fresh Codex thread.

        App Server accepts dynamic tools only on `thread/start`; sending them
        to `thread/resume` is ignored. The caller replays Pupa's transcript (or
        an in-turn continuation prompt) when this returns True.
        """
        if self.surface is not None and self.surface.fingerprint == surface.fingerprint:
            self.surface = surface
            return False
        old = self.client
        self.client = None
        self.codex_thread_id = None
        self.current_turn_id = None
        if old is not None:
            await old.close()
        await self.connect(surface)
        return True

    async def start_turn(
        self,
        *,
        text: str,
        images: list[dict[str, Any]],
        model: str | None,
        effort: str | None,
        state: dict[str, Any] | None,
        transcript: str | None = None,
        run_id: str | None = None,
        preserve_request: bool = False,
    ) -> None:
        if self.client is None or not self.codex_thread_id:
            raise AppServerError("Codex session is not connected")
        self.scope = native_scope(state)
        self.auto_approve_commands = self.auto_approve_commands or auto_approve(state)
        if not preserve_request:
            self._request_text = text
            self._request_transcript = transcript if transcript is not None else text
            self._request_images = list(images)
            self._completed_frontend.clear()
        if run_id is not None:
            self.open_run(run_id)
        self.turn_active = True
        self._turn_done.clear()
        user_input: list[dict[str, Any]] = []
        if text:
            user_input.append({"type": "text", "text": text})
        user_input.extend(images)
        params: dict[str, Any] = {
            "threadId": self.codex_thread_id,
            "input": user_input or [{"type": "text", "text": ""}],
            "cwd": codex_workspace(),
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "sandboxPolicy": turn_sandbox(self.scope, codex_workspace()),
        }
        if model:
            params["model"] = model
        if effort:
            params["effort"] = effort
        result = await self.client.request("turn/start", params, timeout=60.0)
        turn = result.get("turn") or {}
        if isinstance(turn, dict) and turn.get("id"):
            self.current_turn_id = str(turn["id"])

    def arm_surface_continuation(self) -> None:
        """Prevent a fast old turn from closing the resume HTTP run."""
        self._continuing = True
        self._turn_done.clear()

    async def continue_with_surface(
        self,
        surface: ToolSurface,
        *,
        model: str | None,
        effort: str | None,
        state: dict[str, Any] | None,
    ) -> None:
        """Interrupt a narrow turn and continue on a widened fresh thread."""
        if self.client is None:
            raise AppServerError("Codex session is not connected")
        if self.turn_active and not self._continuing:
            self.arm_surface_continuation()
        try:
            if self.turn_active and self.current_turn_id:
                try:
                    await self.client.request(
                        "turn/interrupt",
                        {"threadId": self.codex_thread_id, "turnId": self.current_turn_id},
                        timeout=10.0,
                    )
                except AppServerError:
                    # The tool result can finish a short turn while the
                    # interrupt request is in flight. Its completion event is
                    # authoritative; other failures still surface.
                    if not self._turn_done.is_set():
                        raise
                await asyncio.wait_for(self._turn_done.wait(), timeout=10.0)
            self._continuing = False
            await self.reload_surface(surface)
            completed = "\n".join(self._completed_frontend) or "- none"
            transcript = self._request_transcript or self._request_text
            await self.start_turn(
                text=(
                    "Pupa interrupted the prior Codex turn only to refresh its dynamic "
                    "tools. Continue the original request without repeating completed "
                    "tool calls.\n\nConversation transcript:\n"
                    f"{transcript}\n\nCompleted frontend tool results:\n{completed}"
                ),
                images=self._request_images,
                model=model,
                effort=effort,
                state=state,
                preserve_request=True,
            )
        except Exception:
            self._continuing = False
            raise

    async def resolve_tool_results(self, results: list[dict[str, Any]]) -> None:
        by_id = {row.get("toolCallId"): row.get("content", "") for row in results}
        for call_id in list(self.active_batch):
            pending = self.pending_tools.get(call_id)
            if pending is None or pending.future.done():
                continue
            if call_id in by_id:
                ok, content = True, by_id[call_id]
            else:
                ok, content = False, "missing_tool_result"
            args = json.dumps(pending.arguments, sort_keys=True, default=str)
            self._completed_frontend.append(
                f"- {pending.name}({args}) => {content}"
            )
            pending.future.set_result((ok, content))
        self.active_batch.clear()
        self.touch()
        # Give JSON-RPC request tasks a scheduling turn to write their responses.
        await asyncio.sleep(0)

    async def resolve_approval(self, allow: bool, always: bool = False) -> None:
        future = self.pending_approval
        self.pending_approval = None
        if allow and always:
            self.auto_approve_commands = True
        if future is not None and not future.done():
            future.set_result((allow, always))
        self.touch()

    async def on_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "item/tool/call":
            return await self._dynamic_tool(params)
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
        }:
            return await self._approval(method, params)
        if method == "item/tool/requestUserInput":
            # Developer instructions steer Codex toward plain chat. Empty answers
            # are the fail-safe for a model that still invokes the native dialog.
            return {"answers": {}}
        if method == "mcpServer/elicitation/request":
            return {"action": "decline", "content": None}
        raise AppServerError(f"unsupported Codex App Server request {method!r}")

    async def _dynamic_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        namespace = params.get("namespace")
        name = str(params.get("tool") or "")
        call_id = str(params.get("callId") or params.get("itemId") or uuid.uuid4())
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        surface = self.surface
        if surface is None:
            return _tool_response(False, "Pupa tool surface is unavailable")
        if namespace == MCP_NAMESPACE:
            self._display_tool(call_id, name, arguments)
            ok, content = await invoke_mcp(surface, name, arguments)
            return _tool_response(ok, content)
        if namespace != FRONTEND_NAMESPACE or name not in surface.frontend:
            return _tool_response(False, f"Unknown Pupa frontend tool: {name}")

        future = asyncio.get_running_loop().create_future()
        self.pending_tools[call_id] = PendingTool(call_id, name, arguments, future)
        self._batch_calls.append({"id": call_id, "name": name, "args": arguments})
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._flush_frontend_batch())
        try:
            ok, content = await future
            return _tool_response(bool(ok), str(content))
        finally:
            self.pending_tools.pop(call_id, None)

    async def _flush_frontend_batch(self) -> None:
        await asyncio.sleep(0.01)
        calls, self._batch_calls = self._batch_calls, []
        if not calls:
            return
        self.active_batch = {str(call["id"]) for call in calls}
        for call in calls:
            for event in events.tool_call_events(
                str(call["id"]), str(call["name"]), call.get("args") or {}
            ):
                self.emit(event)
        self.emit(events.on_interrupt(calls))
        self.end_http_run()

    def _approval_within_ceiling(self, method: str, params: dict[str, Any]) -> bool:
        # Codex sends these requests when it wants to cross its current
        # sandbox. A chat approval must never silently widen Pupa's hard
        # read/workspace ceiling; only the explicit `full` scope allows an
        # unsandboxed operation to reach the user for approval.
        if self.scope != "full":
            return False
        return method != "item/permissions/requestApproval"

    async def _approval(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._approval_within_ceiling(method, params):
            return _approval_response(method, False, False, params)
        async with self.approval_lock:
            if self.auto_approve_commands:
                return _approval_response(method, True, True, params)
            future = asyncio.get_running_loop().create_future()
            self.pending_approval = future
            description = _approval_description(method, params)
            message_id = f"codex_approval_{uuid.uuid4().hex}"
            for event in events.text_events(
                message_id,
                f"Codex would like permission to {description}. Reply yes, no, or always.",
            ):
                self.emit(event)
            self.end_http_run()
            allow, always = await future
            return _approval_response(method, bool(allow), bool(always), params)

    def _display_tool(self, call_id: str, name: str, arguments: Any) -> None:
        if call_id in self.seen_items:
            return
        self.seen_items.add(call_id)
        for event in events.tool_call_events(call_id, name, arguments):
            self.emit(event)

    async def on_notification(self, method: str, params: dict[str, Any]) -> None:
        if self.disposed:
            return
        if method == "turn/started":
            turn = params.get("turn") or {}
            if isinstance(turn, dict) and turn.get("id"):
                self.current_turn_id = str(turn["id"])
            return
        if method == "item/agentMessage/delta":
            message_id = str(params.get("itemId") or uuid.uuid4())
            if message_id not in self.open_text:
                self.open_text.add(message_id)
                self.emit(
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant",
                    )
                )
            delta = params.get("delta")
            if delta:
                self.emit(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=str(delta),
                    )
                )
            return
        if method == "item/started":
            item = params.get("item") or {}
            if isinstance(item, dict):
                self._display_native_item(item)
            return
        if method == "item/completed":
            item = params.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "agentMessage":
                message_id = str(item.get("id") or uuid.uuid4())
                if message_id in self.open_text:
                    self.emit(
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END, message_id=message_id
                        )
                    )
                    self.open_text.discard(message_id)
                elif item.get("text"):
                    for event in events.text_events(message_id, str(item["text"])):
                        self.emit(event)
            return
        if method == "turn/completed":
            self.turn_active = False
            self._close_text()
            self._turn_done.set()
            if self._continuing:
                self._continuing = False
                return
            turn = params.get("turn") or {}
            status = turn.get("status") if isinstance(turn, dict) else None
            status_type = status.get("type") if isinstance(status, dict) else status
            if status_type in {"failed", "error"}:
                error = turn.get("error") if isinstance(turn, dict) else None
                self.end_http_run(error=f"Codex turn failed: {error or status_type}")
            else:
                self.end_http_run()
            return
        if method in {"error", "process/exited"}:
            detail = params.get("message") or params.get("error") or "Codex App Server exited"
            self.turn_active = False
            self.end_http_run(error=str(detail))

    def _display_native_item(self, item: dict[str, Any]) -> None:
        item_type = str(item.get("type") or "")
        if item_type in {"agentMessage", "reasoning", "userMessage", "dynamicToolCall"}:
            return
        call_id = str(item.get("id") or uuid.uuid4())
        if item_type == "commandExecution":
            self._display_tool(call_id, "Codex command", {"command": item.get("command")})
        elif item_type == "fileChange":
            self._display_tool(call_id, "Codex file change", {"changes": item.get("changes")})
        elif item_type == "mcpToolCall":
            name = f"{item.get('server') or 'MCP'}: {item.get('tool') or 'tool'}"
            self._display_tool(call_id, name, item.get("arguments") or {})
        elif item_type == "webSearch":
            self._display_tool(call_id, "Codex web search", {"query": item.get("query")})
        elif item_type == "collabAgentToolCall":
            self._display_tool(call_id, f"Codex {item.get('tool') or 'agent'}", item)
        elif item_type:
            self._display_tool(call_id, f"Codex {item_type}", item)

    async def dispose(self, *, notify: bool = True) -> None:
        if self.disposed:
            return
        if notify and self.current_run_id:
            self.emit(events.run_error("the Codex session ended before the turn completed"))
            self.queue.put_nowait(ERROR)
        self.disposed = True
        for pending in list(self.pending_tools.values()):
            if not pending.future.done():
                pending.future.set_result((False, "session_disposed"))
        if self.pending_approval is not None and not self.pending_approval.done():
            self.pending_approval.set_result((False, False))
        if self._batch_task is not None and not self._batch_task.done():
            self._batch_task.cancel()
        if self.client is not None:
            await self.client.close()


def _tool_response(ok: bool, content: str) -> dict[str, Any]:
    return {
        "contentItems": [{"type": "inputText", "text": content}],
        "success": ok,
    }


def _approval_description(method: str, params: dict[str, Any]) -> str:
    if method == "item/commandExecution/requestApproval":
        network = params.get("networkApprovalContext") or {}
        if isinstance(network, dict) and network.get("host"):
            return f"access {network.get('protocol') or 'the network'} host {network['host']}"
        command = params.get("command")
        return f"run `{command}`" if command else "run a command"
    if method == "item/fileChange/requestApproval":
        return params.get("reason") or "modify files"
    return params.get("reason") or "expand its sandbox permissions"


def _approval_response(
    method: str,
    allow: bool,
    always: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    if method == "item/permissions/requestApproval":
        return {
            "permissions": params.get("permissions") if allow else {},
            "scope": "session" if always else "turn",
        }
    return {"decision": "acceptForSession" if allow and always else "accept" if allow else "decline"}


_REGISTRY: dict[str, LiveSession] = {}
_THREAD_IDS: dict[str, str] = {}
_THREAD_SURFACES: dict[str, str] = {}


def get(thread_id: str) -> LiveSession | None:
    return _REGISTRY.get(thread_id)


def create(thread_id: str, mcp: Any) -> LiveSession:
    session = LiveSession(thread_id=thread_id, mcp=mcp)
    _REGISTRY[thread_id] = session
    return session


def remembered_thread_id(thread_id: str) -> str | None:
    return _THREAD_IDS.get(thread_id)


def remembered_surface(thread_id: str) -> str | None:
    return _THREAD_SURFACES.get(thread_id)


def remember_thread_id(
    thread_id: str,
    codex_thread_id: str,
    surface_fingerprint: str | None = None,
) -> None:
    _THREAD_IDS[thread_id] = codex_thread_id
    if surface_fingerprint is not None:
        _THREAD_SURFACES[thread_id] = surface_fingerprint


def forget_thread_id(thread_id: str) -> None:
    _THREAD_IDS.pop(thread_id, None)
    _THREAD_SURFACES.pop(thread_id, None)


async def remove(thread_id: str, session: LiveSession | None = None, *, notify: bool = True) -> None:
    current = _REGISTRY.get(thread_id)
    if session is not None and current is not session:
        await session.dispose(notify=notify)
        return
    _REGISTRY.pop(thread_id, None)
    if current is not None:
        await current.dispose(notify=notify)


async def attach(session: LiveSession):
    """Drain one ordered AG-UI stream, handing the queue to a newer attach."""
    async for item in single_consumer_events(
        session, logger=logger, label="codex harness"
    ):
        if item is BOUNDARY:
            session.touch()
            return
        if item is ERROR:
            await remove(session.thread_id, session, notify=False)
            return
        yield item


def note_reattach(thread_id: str) -> None:
    session = _REGISTRY.get(thread_id)
    if session is not None:
        session.touch()


async def sweep_idle(timeout: float | None = None) -> int:
    wall = _idle_timeout() if timeout is None else timeout
    now = time.monotonic()
    stale = [
        thread_id
        for thread_id, session in _REGISTRY.items()
        if now - session.last_activity > wall
    ]
    for thread_id in stale:
        logger.info("codex harness: evicting idle session thread_id=%s", thread_id)
        await remove(thread_id, notify=False)
    return len(stale)


async def shutdown_all() -> None:
    for thread_id in list(_REGISTRY):
        await remove(thread_id, notify=False)
