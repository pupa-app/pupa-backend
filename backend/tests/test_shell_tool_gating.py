"""Env-gating contract for the optional `shell` backend tool.

`ShellToolMiddleware` is wired through the same `BackendToolSpec` registry
that gates `tavily_search` by `TAVILY_API_KEY`. Three contracts matter:

1. **Default-off**: without `SHELL_TOOL_ENABLED`, `build_middlewares()`
   returns nothing for the shell spec and the discovery endpoint reports
   the spec as `enabled_by_env=False` (iOS Settings greys it out).

2. **Enabled**: with `SHELL_TOOL_ENABLED` set, `build_middlewares()`
   materialises `[ShellApprovalMiddleware, ShellToolMiddleware]` — approval
   is **always** paired with the shell tool so unattended execution is
   never the backend's default.  Users opt out via the iOS Settings toggle,
   which surfaces as `state["shell_approval_disabled"]=True` per turn.

3. **Workspace pinning** still threads through `SHELL_TOOL_WORKSPACE`.
"""



import os

import pytest
from langchain.agents.middleware.shell_tool import ShellToolMiddleware

from pupa_backend.harnesses.langgraph.backend_tools import BACKEND_TOOLS, build_middlewares, enabled_specs
from pupa_backend.harnesses.langgraph.shell_approval import ShellApprovalMiddleware


@pytest.fixture(autouse=True)
def _clear_shell_env(monkeypatch):
    monkeypatch.delenv("SHELL_TOOL_ENABLED", raising=False)
    monkeypatch.delenv("SHELL_TOOL_WORKSPACE", raising=False)
    # The `skills` and `subagents` specs are on by default; disable them here so
    # these shell-scoped assertions about `build_middlewares()` see only the
    # shell spec.
    monkeypatch.setenv("PUPA_SKILLS_DISABLED", "1")
    monkeypatch.setenv("PUPA_SUBAGENTS_DISABLED", "1")


def _shell_spec():
    return next(spec for spec in BACKEND_TOOLS if spec.name == "shell")


def test_shell_spec_registered_with_env_gate():
    spec = _shell_spec()
    assert spec.env_var == "SHELL_TOOL_ENABLED"
    assert spec.middleware_factory is not None
    assert spec.factory is None


def test_shell_disabled_when_env_unset():
    assert _shell_spec().enabled_by_env is False
    assert _shell_spec() not in enabled_specs()
    assert build_middlewares() == []


def test_shell_enabled_pairs_approval_with_tool(monkeypatch):
    monkeypatch.setenv("SHELL_TOOL_ENABLED", "1")
    assert _shell_spec().enabled_by_env is True
    middlewares = build_middlewares()
    assert len(middlewares) == 2
    assert isinstance(middlewares[0], ShellApprovalMiddleware)
    assert isinstance(middlewares[1], ShellToolMiddleware)


def test_shell_workspace_env_threaded(monkeypatch, tmp_path):
    monkeypatch.setenv("SHELL_TOOL_ENABLED", "1")
    monkeypatch.setenv("SHELL_TOOL_WORKSPACE", str(tmp_path))
    middlewares = build_middlewares()
    assert isinstance(middlewares[-1], ShellToolMiddleware)
