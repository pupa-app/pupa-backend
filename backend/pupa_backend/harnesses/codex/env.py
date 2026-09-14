"""Codex harness configuration, sandbox policy, and subscription guard."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from pupa_backend.prompts import SYSTEM_PROMPT


class CodexSubscriptionUnavailable(RuntimeError):
    pass


def codex_binary() -> str:
    configured = (os.getenv("PUPA_CODEX_BIN") or "").strip()
    return configured or shutil.which("codex") or "codex"


def codex_workspace() -> str:
    raw = (os.getenv("PUPA_CODEX_WORKSPACE") or "").strip()
    path = Path(raw).expanduser() if raw else Path.cwd()
    return str(path.resolve())


def child_env() -> dict[str, str]:
    """Minimal child environment; deliberately excludes every API billing key."""
    allowed = {
        "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE",
        "SHELL", "TERM", "TMPDIR", "SSH_AUTH_SOCK",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed and value}
    config_dir = (os.getenv("PUPA_CODEX_CONFIG_DIR") or "").strip()
    if config_dir:
        env["CODEX_HOME"] = str(Path(config_dir).expanduser())
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_API_KEY", None)
    return env


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def native_scope(state: dict[str, Any] | None = None) -> str:
    raw: Any = None
    if isinstance(state, dict):
        raw = state.get("codex_loop_native") or state.get("codexLoopNative")
    raw = raw or os.getenv("PUPA_CODEX_LOOP_NATIVE") or "workspace"
    value = str(raw).strip().lower()
    aliases = {"read-only": "read", "workspace-write": "workspace", "danger-full-access": "full"}
    value = aliases.get(value, value)
    return value if value in {"read", "workspace", "full"} else "workspace"


def auto_approve(state: dict[str, Any] | None = None) -> bool:
    if isinstance(state, dict):
        value = state.get("codex_loop_auto_approve")
        if value is None:
            value = state.get("autoApprove")
        if value is not None:
            return _truthy(value)
    return _truthy(os.getenv("PUPA_CODEX_LOOP_AUTO_APPROVE"))


def thread_sandbox(scope: str) -> str:
    return {
        "read": "read-only",
        "workspace": "workspace-write",
        "full": "danger-full-access",
    }[scope]


def turn_sandbox(scope: str, workspace: str) -> dict[str, Any]:
    if scope == "read":
        return {"type": "readOnly", "networkAccess": False}
    if scope == "full":
        return {"type": "dangerFullAccess"}
    return {
        "type": "workspaceWrite",
        "writableRoots": [workspace],
        "networkAccess": False,
    }


def developer_instructions() -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "You are running inside the Pupa app. Pupa has no native terminal dialog or "
        "question UI. Ask the user in normal assistant chat, or use a frontend tool "
        "advertised by Pupa. Do not use a built-in request-user-input tool."
    )
