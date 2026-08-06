"""Tests for the `claude_code` backend tool (subprocess shell-out + gating).

No real `claude` CLI is invoked — `asyncio.create_subprocess_exec` is
monkeypatched with a fake process so we pin the argv we build, the JSON parsing,
and the error/timeout paths. The registry test pins the on-by-default opt-out
gate.
"""

import asyncio
import json

import pytest

import pupa_backend.harnesses.langgraph.claude_code_tool as claude_code_tool
from pupa_backend.harnesses.langgraph.claude_code_tool import _build_argv, _format_result, claude_code


class _FakeProc:
    """Minimal stand-in for an asyncio subprocess."""

    def __init__(self, stdout=b"", stderr=b"", returncode=0, hang=False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(60)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _patch_exec(monkeypatch, proc):
    """Patch create_subprocess_exec to capture argv/kwargs and return `proc`."""
    captured = {}

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(claude_code_tool.asyncio, "create_subprocess_exec", fake_exec)
    return captured


# ---- argv construction (pure) ----------------------------------------------

def test_argv_plan_mode_is_read_only():
    argv = _build_argv("do a thing", "plan", None)
    assert "-p" in argv and "do a thing" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--permission-mode") + 1] == "plan"
    # Read-only belt-and-suspenders tool allowlist present in plan mode.
    assert "--allowedTools" in argv
    assert {"Read", "Grep", "Glob"}.issubset(set(argv))


def test_argv_edit_mode_drops_read_only_allowlist():
    argv = _build_argv("fix the bug", "edit", None)
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert "--allowedTools" not in argv


def test_argv_resume_appends_session():
    argv = _build_argv("continue", "plan", "sess-123")
    assert argv[argv.index("--resume") + 1] == "sess-123"


def test_argv_respects_bin_and_model(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", "/opt/claude")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-4-8")
    argv = _build_argv("x", "plan", None)
    assert argv[0] == "/opt/claude"
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"


# ---- result formatting (pure) ----------------------------------------------

def test_format_result_extracts_text_and_session():
    out = _format_result(json.dumps({
        "result": "Here is the summary.",
        "session_id": "abc",
        "total_cost_usd": 0.01,
        "is_error": False,
    }))
    assert "Here is the summary." in out
    assert "abc" in out
    assert "0.01" in out


def test_format_result_flags_error():
    out = _format_result(json.dumps({"result": "boom", "is_error": True}))
    assert "error" in out.lower()


def test_format_result_falls_back_on_bad_json():
    assert _format_result("not json") == "not json"


# ---- full tool, subprocess monkeypatched -----------------------------------

async def test_tool_returns_result_string(monkeypatch):
    proc = _FakeProc(
        stdout=json.dumps({
            "result": "All done.",
            "session_id": "s1",
            "total_cost_usd": 0.02,
            "is_error": False,
        }).encode(),
    )
    captured = _patch_exec(monkeypatch, proc)

    result = await claude_code.ainvoke({"prompt": "summarize the README"})

    assert "All done." in result
    assert "s1" in result
    assert captured["argv"][0:2] == ["claude", "-p"]
    assert captured["kwargs"]["env"] is not None  # minimal env passed, not inherited


async def test_tool_nonzero_exit_returns_error_string(monkeypatch):
    proc = _FakeProc(stdout=b"", stderr=b"boom on the host", returncode=1)
    _patch_exec(monkeypatch, proc)

    result = await claude_code.ainvoke({"prompt": "x"})

    assert "failed" in result.lower()
    assert "boom on the host" in result


async def test_tool_missing_binary_returns_error_string(monkeypatch):
    async def boom(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(claude_code_tool.asyncio, "create_subprocess_exec", boom)

    result = await claude_code.ainvoke({"prompt": "x"})
    assert "not found" in result.lower()


async def test_tool_timeout_kills_and_reports(monkeypatch):
    proc = _FakeProc(hang=True)
    _patch_exec(monkeypatch, proc)
    monkeypatch.setenv("CLAUDE_CODE_TIMEOUT", "0.05")

    result = await claude_code.ainvoke({"prompt": "x"})

    assert "timed out" in result.lower()
    assert proc.killed is True


# ---- env: subscription-billed by default -----------------------------------

def test_build_env_subscription_only_by_default(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_ALLOW_API_BILLING", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_PASS_ENV", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-orchestrator")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-x")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = claude_code_tool._build_env()
    # Orchestrator creds must NOT reach the spawned sub-agent → it uses the sub.
    for var in claude_code_tool._CLAUDE_CRED_VARS:
        assert var not in env, f"{var} leaked into the sub-agent env"
    assert env.get("PATH") == "/usr/bin"


def test_build_env_allow_api_billing_forwards_creds(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ALLOW_API_BILLING", "1")
    monkeypatch.delenv("CLAUDE_CODE_PASS_ENV", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    env = claude_code_tool._build_env()
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant-x"


def test_build_env_pass_env_still_strips_creds_by_default(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_PASS_ENV", "1")
    monkeypatch.delenv("CLAUDE_CODE_ALLOW_API_BILLING", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    env = claude_code_tool._build_env()
    assert "ANTHROPIC_API_KEY" not in env


# ---- registry gating -------------------------------------------------------

def test_spec_enabled_by_default(monkeypatch):
    monkeypatch.delenv("PUPA_CLAUDE_CODE_DISABLED", raising=False)
    from pupa_backend.harnesses.langgraph.backend_tools import enabled_specs

    names = {s.name for s in enabled_specs()}
    assert "claude_code" in names


def test_spec_opt_out_disables(monkeypatch):
    monkeypatch.setenv("PUPA_CLAUDE_CODE_DISABLED", "1")
    from pupa_backend.harnesses.langgraph.backend_tools import enabled_specs

    names = {s.name for s in enabled_specs()}
    assert "claude_code" not in names
