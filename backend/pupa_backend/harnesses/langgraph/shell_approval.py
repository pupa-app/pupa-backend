"""Pre-execution approval gate for the shell tool.

`ShellApprovalMiddleware` pauses before every `shell` command and asks the
user to approve it. The approval request is surfaced to the client as a
regular `frontend_tool_calls` interrupt — the same channel used by
`ask_user_questions` and every other frontend tool. The iOS client
registers a `request_shell_approval` handler in its ToolRegistry, shows
an Approve/Deny card, and resumes with a standard
`tool_results=[{toolCallId, content}]` payload where `content` is a JSON
object `{"approved": bool, "remember": bool}`.

**Why the interrupt lives in `after_model`, not `awrap_tool_call`.**
LangGraph's `ToolNode` runs every pending tool call in parallel — one
`awrap_tool_call` task per call. If the interrupt fired there, a model
turn with two `shell` calls would raise two `interrupt()`s *simultaneously*
(one per task), leaving two pending interrupts at the same checkpoint.
Resuming with a plain `Command(resume=...)` then crashes with
``RuntimeError: When there are multiple pending interrupts, you must
specify the interrupt id when resuming``.

So approval is split across two hooks:

- **`after_model`** runs once per model turn (its own graph node), collects
  every `shell` call that still needs approval, and fires a *single*
  batched interrupt listing them all — mirroring
  `CustomCopilotKitMiddleware.after_model`. On resume it
  records the per-call decisions into `state["shell_approval_decisions"]`
  (and updates the remembered-allowlist for "always allow"). Because the
  interrupt lives in its own node, resuming re-runs only this hook — the
  model is never re-invoked.

- **`awrap_tool_call`** stays a pure execution gate: for each `shell`
  call it reads the decision recorded by `after_model` and either runs
  the command (approved / pre-approved / disabled) or returns a denial
  `ToolMessage` without executing. No interrupt happens here, so the
  parallel-task case is safe.

**Per-thread allowlist.**  When the user ticks "always allow", the
command string is stored in an in-memory per-thread dict keyed by the
LangGraph `thread_id` from `get_config()`.  That dict lives on the
middleware instance, which is created once at `build_graph()` time and
shared across all requests.  State persists for the lifetime of the
backend process; restarting the backend clears it (acceptable for the
dev use case this feature targets).  Cross-restart persistence is a
Phase C nice-to-have.

The "allow once / allow always" semantic is deliberately command-string-
exact: `ls -la /tmp` and `ls -la /home` are treated as separate commands.
Pattern-based allowlists are a future extension.
"""



import json
import logging
from typing import Any, Awaitable, Callable, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ToolMessage,
)
from langchain_core.messages import AIMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from pupa_backend.agui.tool_results import parse_tool_results

logger = logging.getLogger("uvicorn.error")


class ShellApprovalState(AgentState):
    """State extension for the shell approval gate.

    - `shell_approved_commands` is the per-thread allowlist of already-approved
      command strings (the "Always allow this session" path).  Stored here so
      `ag_ui_langgraph.prepare_stream` doesn't strip it through the input
      schema filter (same reason `ToolGatingState` declares `disabled_tools`).

    - `shell_approval_disabled` is a per-turn client override — when the user
      toggles "Require shell approval" off in iOS Settings, the next turn's
      `RunAgentInput.state` carries `shell_approval_disabled: True` and the
      middleware bypasses the interrupt entirely for that turn.

    - `shell_approval_decisions` is the internal hand-off from `after_model`
      (which runs the batched interrupt) to `awrap_tool_call` (which executes
      or denies each call). Keyed by tool-call id → `{approved, remember}`.
      Never sent by the client; declared here so it survives the interrupt
      checkpoint and the model→tools state transition.
    """

    shell_approved_commands: NotRequired[list[str]]
    shell_approval_disabled: NotRequired[bool]
    shell_approval_decisions: NotRequired[dict[str, dict[str, bool]]]


class ShellApprovalMiddleware(AgentMiddleware):
    """Pause-before-execute gate for the `shell` tool.

    Fires a single batched `request_shell_approval` frontend-tool interrupt in
    `after_model` covering every shell command in the turn that isn't already
    pre-approved, then enforces the per-call decision in `awrap_tool_call`.
    """

    state_schema = ShellApprovalState

    def __init__(self) -> None:
        self._approved: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _thread_id(self) -> str | None:
        return get_config().get("configurable", {}).get("thread_id")

    def _is_pre_approved(self, command: str) -> bool:
        tid = self._thread_id()
        return tid is not None and command in self._approved.get(tid, set())

    def _record_approval(self, command: str) -> None:
        tid = self._thread_id()
        if tid is None:
            return
        self._approved.setdefault(tid, set()).add(command)

    @staticmethod
    def _shell_calls(message: Any) -> list[dict]:
        """Return the `shell` tool calls on an AIMessage (empty if none)."""
        if not isinstance(message, AIMessage):
            return []
        tool_calls = getattr(message, "tool_calls", None) or []
        return [c for c in tool_calls if c.get("name") == "shell"]

    # ------------------------------------------------------------------
    # Batched approval interrupt (once per model turn)
    # ------------------------------------------------------------------

    def after_model(
        self,
        state: ShellApprovalState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        # Per-turn client override: skip the interrupt entirely. `awrap_tool_call`
        # also honours this flag and runs the command directly.
        if state.get("shell_approval_disabled"):
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        shell_calls = self._shell_calls(messages[-1])
        if not shell_calls:
            return None

        # Only the calls whose command isn't already remembered need approval.
        # Pre-approved calls fall straight through to `awrap_tool_call`.
        need_approval = [
            c for c in shell_calls
            if not self._is_pre_approved(c.get("args", {}).get("command") or "")
        ]
        if not need_approval:
            return None

        # One interrupt for the whole turn — never one-per-call. This is the
        # crux of the fix: parallel shell calls resolve through a single
        # pending interrupt, so `Command(resume=...)` needs no interrupt id.
        payload = interrupt({
            "frontend_tool_calls": [
                {
                    "id": c.get("id"),
                    "name": "request_shell_approval",
                    "args": {"command": c.get("args", {}).get("command") or ""},
                }
                for c in need_approval
            ]
        })

        results = parse_tool_results(payload)
        results_by_id = {r["toolCallId"]: r["content"] for r in results}

        decisions: dict[str, dict[str, bool]] = {}
        for call in need_approval:
            tc_id = call.get("id") or ""
            command = call.get("args", {}).get("command") or ""
            raw_content = results_by_id.get(tc_id, "{}")
            try:
                decision: dict[str, Any] = (
                    json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                )
            except json.JSONDecodeError:
                decision = {}

            approved = bool(decision.get("approved", False))
            remember = bool(decision.get("remember", False))
            decisions[tc_id] = {"approved": approved, "remember": remember}

            if approved and remember:
                self._record_approval(command)
                logger.info("[shell_approval] approved + remembered: %r (threadId=%s)", command, self._thread_id())
            elif approved:
                logger.info("[shell_approval] approved once: %r (threadId=%s)", command, self._thread_id())
            else:
                logger.info("[shell_approval] denied: %r (threadId=%s)", command, self._thread_id())

        # Merge with any decisions already in state so a mixed turn where some
        # calls were pre-approved (and thus absent from `need_approval`) keeps
        # a coherent map. Decisions are keyed by unique tool-call id.
        existing = state.get("shell_approval_decisions") or {}
        return {"shell_approval_decisions": {**existing, **decisions}}

    async def aafter_model(
        self,
        state: ShellApprovalState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    # ------------------------------------------------------------------
    # Per-call execution gate (runs inside ToolNode, possibly in parallel)
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        if request.tool_call.get("name") != "shell":
            return await handler(request)

        command: str = request.tool_call.get("args", {}).get("command") or ""
        tc_id: str = request.tool_call.get("id", "")

        # Client-side global override: iOS Settings sheet toggle "Require shell
        # approval" off → state["shell_approval_disabled"]=True per turn.
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict) and state.get("shell_approval_disabled"):
            logger.info("[shell_approval] disabled by client: %r (threadId=%s)", command, self._thread_id())
            return await handler(request)

        if self._is_pre_approved(command):
            logger.info("[shell_approval] pre-approved: %r (threadId=%s)", command, self._thread_id())
            return await handler(request)

        # Decision was made by `after_model`'s batched interrupt and recorded
        # in state. A missing entry is treated as denial — fail closed.
        decisions = state.get("shell_approval_decisions") or {} if isinstance(state, dict) else {}
        approved = bool(decisions.get(tc_id, {}).get("approved", False))

        if not approved:
            return ToolMessage(
                content=(
                    "User denied this command at this time. "
                ),
                tool_call_id=tc_id,
                name="shell",
            )

        return await handler(request)
