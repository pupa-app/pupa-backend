"""Backend `claude_code` tool — delegate a job to a full Claude Code agent.

Shells out to the `claude` CLI in headless mode (`claude -p ... --output-format
json`) so the main Pupa agent can hand off a self-contained coding/research
task to a fresh Claude Code agent running in its own context and tool loop, then
get back a single synthesized result. This keeps long, multi-step sub-jobs out
of the orchestrator's conversation context — the same motivation as the
in-process `subagents` (`task`) tool, but with a *separate process* that has its
own filesystem-aware tool loop instead of sharing this agent's tools.

One-shot transport (vs a persistent ACP/JSON-RPC transport): each call spawns
`claude` once, waits for the final JSON
object, and exits. `--resume <session_id>` gives multi-turn continuity without a
long-lived process, since the CLI persists sessions to disk. Swap to a
persistent transport later behind this same `BackendToolSpec` if live streaming
to the client is ever wanted.

Safety posture:
  - **Read-only by default.** `mode="plan"` runs `--permission-mode plan` plus a
    belt-and-suspenders `--allowedTools Read Grep Glob`, so the spawned agent
    cannot edit files or run commands. The caller opts into writes per call with
    `mode="edit"` (`--permission-mode acceptEdits`).
  - **Subscription billing by default.** The spawned process does NOT inherit the
    backend's full environment — only PATH/HOME and a few locale vars — and
    crucially **does not** carry any API/Bedrock credential, so the sub-agent
    authenticates with the host's Claude Pro/Max subscription login. This is a
    separate process from the orchestrator, so the backend keeping its own
    `ANTHROPIC_API_KEY` (e.g. for the anthropic LLM provider) does not leak into
    the sub-agent. To restore the old per-token API/Bedrock billing (e.g. no
    subscription on the host), set `CLAUDE_CODE_ALLOW_API_BILLING=1`, which
    forwards `_CLAUDE_CRED_VARS` again. `CLAUDE_CODE_PASS_ENV=1` forwards the whole
    environment (minus `shell_env_excluded()`) but still strips the credential vars
    unless `CLAUDE_CODE_ALLOW_API_BILLING=1` is also set.

Default ON; opt out with `PUPA_CLAUDE_CODE_DISABLED=1`. Wired through
`backend_tools.py` like every other optional backend capability.

Env knobs:
  - ``CLAUDE_CODE_BIN``       — binary to run (default ``claude``).
  - ``CLAUDE_CODE_MODEL``     — model passed via ``--model`` (default: unset →
                                claude's own default).
  - ``CLAUDE_CODE_WORKSPACE`` — working dir for the spawned agent (default:
                                backend process cwd). Read-only counterpart to
                                ``SHELL_TOOL_WORKSPACE``.
  - ``CLAUDE_CODE_TIMEOUT``   — seconds before the job is killed (default 900).
  - ``CLAUDE_CODE_MAX_TURNS`` — ``--max-turns`` cap (default 30).
  - ``CLAUDE_CODE_PASS_ENV``  — ``1`` to forward the full backend env (cred vars
                                still stripped unless API billing is opted in).
  - ``CLAUDE_CODE_ALLOW_API_BILLING`` — ``1`` to forward API/Bedrock creds again
                                (opt back into per-token billing; default is
                                subscription-only).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from langchain_core.tools import tool

# Credential env vars `claude` itself needs even in the minimal-env case. The
# spawned agent authenticates with its own provider creds; these are forwarded
# explicitly so a minimal env doesn't break it.
_CLAUDE_CRED_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_USE_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_PROFILE",
)

_DEFAULT_TIMEOUT = 900.0
_DEFAULT_MAX_TURNS = 30


def _build_env() -> dict[str, str]:
    """Environment for the spawned `claude` process.

    **Subscription-billed by default**: API/Bedrock credential vars are *not*
    forwarded, so the sub-agent authenticates with the host's Claude Pro/Max login
    rather than billing per-token. This process has an explicit env (it does not
    inherit `os.environ` in the default path), so the orchestrator keeping its own
    `ANTHROPIC_API_KEY` doesn't reach the sub-agent. `CLAUDE_CODE_ALLOW_API_BILLING=1`
    forwards `_CLAUDE_CRED_VARS` again. `CLAUDE_CODE_PASS_ENV=1` forwards the full
    environment minus everything `shell_env_excluded()` withholds (secret-shaped
    names plus the operator's `SHELL_ENV_EXCLUDE`) — and still strips the
    credential vars unless API billing is opted in.
    """
    from pupa_backend.auth.devices import truthy

    allow_api = truthy(os.getenv("CLAUDE_CODE_ALLOW_API_BILLING"))

    if os.getenv("CLAUDE_CODE_PASS_ENV"):
        from pupa_backend.harnesses.langgraph.backend_tools import shell_env_filter

        creds = set(_CLAUDE_CRED_VARS)
        excluded = shell_env_filter()

        def _forward(name: str) -> bool:
            # The credential vars are the point of `allow_api` — decide them
            # first, or the generic secret denylist would drop the very keys
            # the opt-in exists to pass through and the sub-agent would get
            # ANTHROPIC_BASE_URL with nothing to authenticate with.
            if name in creds:
                return allow_api
            return not excluded(name)

        return {k: v for k, v in os.environ.items() if _forward(k)}

    # USER/LOGNAME are required for macOS Keychain to resolve the login keychain
    # where the CLI's OAuth login (subscription auth) is stored — without them a
    # subscription-authed claude reports "Not logged in". Not secrets.
    keys = ["PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL"]
    if allow_api:
        keys += list(_CLAUDE_CRED_VARS)
    env: dict[str, str] = {}
    for key in keys:
        val = os.getenv(key)
        if val is not None:
            env[key] = val
    return env


def _build_argv(prompt: str, mode: str, resume_session_id: str | None) -> list[str]:
    """Assemble the `claude` CLI argv for a one-shot headless run."""
    binary = os.getenv("CLAUDE_CODE_BIN", "claude")
    permission_mode = "acceptEdits" if mode == "edit" else "plan"
    max_turns = os.getenv("CLAUDE_CODE_MAX_TURNS", str(_DEFAULT_MAX_TURNS))

    argv: list[str] = [
        binary,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        permission_mode,
        "--max-turns",
        str(max_turns),
    ]
    # Read-only belt-and-suspenders: even though plan mode blocks writes, cap the
    # tool surface to read-only tools. Omitted in edit mode so the agent can act.
    if permission_mode == "plan":
        argv += ["--allowedTools", "Read", "Grep", "Glob"]

    model = os.getenv("CLAUDE_CODE_MODEL")
    if model:
        argv += ["--model", model]

    if resume_session_id:
        argv += ["--resume", resume_session_id]

    return argv


def _format_result(stdout: str) -> str:
    """Parse the CLI's JSON envelope into a compact string for the model.

    `claude -p --output-format json` emits one JSON object with at least
    `result`, `session_id`, `total_cost_usd`, `is_error`. We return the result
    text plus a footer carrying the session id (so the caller can `--resume`)
    and cost. Falls back to the raw stdout if it isn't the expected shape.
    """
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return stdout.strip() or "claude_code returned no output."

    if not isinstance(data, dict):
        return stdout.strip()

    result = data.get("result")
    session_id = data.get("session_id")
    cost = data.get("total_cost_usd")
    is_error = bool(data.get("is_error"))

    body = result if isinstance(result, str) and result.strip() else json.dumps(data)
    footer_bits = []
    if session_id:
        footer_bits.append(f"session_id={session_id} (pass as resume_session_id to continue)")
    if cost is not None:
        footer_bits.append(f"cost=${cost}")
    footer = ("\n\n— " + "; ".join(footer_bits)) if footer_bits else ""
    prefix = "claude_code reported an error:\n\n" if is_error else ""
    return f"{prefix}{body}{footer}"


@tool
async def claude_code(
    prompt: str,
    mode: str = "plan",
    resume_session_id: str | None = None,
) -> str:
    """Delegate a self-contained coding or research job to a full Claude Code agent.

    Spawns a fresh `claude` agent that runs in its own context with its own
    filesystem-aware tool loop, then returns a single synthesized result. Use
    this to offload heavy, multi-step work (investigate a codebase, draft an
    implementation, run an analysis) without crowding your own context. Write a
    fully self-contained prompt — the sub-agent sees nothing of this
    conversation.

    Args:
        prompt: The complete task for the sub-agent. Be specific; it has no
            access to this conversation's history.
        mode: "plan" (default) runs read-only — the sub-agent can read and search
            files but cannot edit them or run commands. "edit" lets it modify
            files (acceptEdits). Only use "edit" when changes are intended.
        resume_session_id: Optional. The session_id returned by a previous
            claude_code call, to continue that sub-agent's session.

    Returns:
        The sub-agent's result text, with a footer carrying its session_id (for
        follow-ups) and cost. Errors are returned as readable strings.
    """
    argv = _build_argv(prompt, mode, resume_session_id)
    cwd = os.getenv("CLAUDE_CODE_WORKSPACE") or None
    try:
        timeout = float(os.getenv("CLAUDE_CODE_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    except (ValueError, TypeError):
        timeout = _DEFAULT_TIMEOUT

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=_build_env(),
        )
    except FileNotFoundError:
        binary = argv[0]
        return (
            f"claude_code: '{binary}' not found on PATH. Install the Claude Code "
            "CLI on the backend host or set CLAUDE_CODE_BIN to its full path."
        )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"claude_code timed out after {timeout:.0f}s and was terminated."

    stdout = (stdout_b or b"").decode("utf-8", "replace")
    stderr = (stderr_b or b"").decode("utf-8", "replace")

    if proc.returncode != 0:
        tail = stderr.strip()[-2000:] or stdout.strip()[-2000:] or "(no output)"
        return f"claude_code failed (exit {proc.returncode}):\n{tail}"

    return _format_result(stdout)


def build_claude_code_tool() -> Any:
    """Return the `claude_code` LangChain tool (factory entrypoint for the spec)."""
    return claude_code
