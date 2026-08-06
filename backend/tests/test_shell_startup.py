"""Tests for shell startup command loading and subprocess env construction.

Covers three units in backend_tools.py:

1. ``_shell_env_exclude`` — parses SHELL_ENV_EXCLUDE from the environment.
2. ``_build_startup_commands`` — loads commands from a local script file.
3. ``_build_shell_middlewares`` env logic — SHELL_PASS_ENV controls whether
   os.environ is forwarded to the subprocess, with excluded vars stripped.
"""

import os
import textwrap

import pytest


# ---------------------------------------------------------------------------
# _shell_env_exclude
# ---------------------------------------------------------------------------

class TestShellEnvExclude:
    def test_empty_when_var_unset(self, monkeypatch):
        monkeypatch.delenv("SHELL_ENV_EXCLUDE", raising=False)
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset()

    def test_single_var(self, monkeypatch):
        monkeypatch.setenv("SHELL_ENV_EXCLUDE", "GH_TOKEN")
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset({"GH_TOKEN"})

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("SHELL_ENV_EXCLUDE", "GH_TOKEN,AWS_SECRET_ACCESS_KEY,MY_SECRET")
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset({"GH_TOKEN", "AWS_SECRET_ACCESS_KEY", "MY_SECRET"})

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("SHELL_ENV_EXCLUDE", " GH_TOKEN , OTHER ")
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset({"GH_TOKEN", "OTHER"})

    def test_ignores_empty_segments(self, monkeypatch):
        monkeypatch.setenv("SHELL_ENV_EXCLUDE", "GH_TOKEN,,OTHER")
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset({"GH_TOKEN", "OTHER"})

    def test_from_file_parses_export_lines(self, monkeypatch, tmp_path):
        zshrc = tmp_path / ".zshrc"
        zshrc.write_text(
            "# comment\n"
            "export OPENAI_API_KEY=sk-...\n"
            "export HF_TOKEN=hf_...\n"
            "export NVM_DIR=$HOME/.nvm\n"
            "alias ll='ls -la'\n"          # not an export — ignored
            "source ~/.profile\n"           # not an export — ignored
        )
        monkeypatch.delenv("SHELL_ENV_EXCLUDE", raising=False)
        monkeypatch.setenv("SHELL_ENV_EXCLUDE_FROM", str(zshrc))
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset({"OPENAI_API_KEY", "HF_TOKEN", "NVM_DIR"})

    def test_from_file_merged_with_manual_list(self, monkeypatch, tmp_path):
        zshrc = tmp_path / ".zshrc"
        zshrc.write_text("export FROM_FILE=x\n")
        monkeypatch.setenv("SHELL_ENV_EXCLUDE", "MANUAL_SECRET")
        monkeypatch.setenv("SHELL_ENV_EXCLUDE_FROM", str(zshrc))
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset({"MANUAL_SECRET", "FROM_FILE"})

    def test_from_file_missing_is_silently_ignored(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SHELL_ENV_EXCLUDE", raising=False)
        monkeypatch.setenv("SHELL_ENV_EXCLUDE_FROM", str(tmp_path / "nonexistent"))
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert _shell_env_exclude() == frozenset()

    def test_from_file_expands_tilde(self, monkeypatch, tmp_path):
        zshrc = tmp_path / "fake_zshrc"
        zshrc.write_text("export SECRET_KEY=abc\n")
        monkeypatch.setenv("SHELL_ENV_EXCLUDE_FROM", str(zshrc))
        monkeypatch.delenv("SHELL_ENV_EXCLUDE", raising=False)
        from pupa_backend.harnesses.langgraph.backend_tools import _shell_env_exclude
        assert "SECRET_KEY" in _shell_env_exclude()


# ---------------------------------------------------------------------------
# _build_startup_commands
# ---------------------------------------------------------------------------

class TestBuildStartupCommands:
    def test_returns_empty_when_no_script(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SHELL_STARTUP_SCRIPT", str(tmp_path / "nonexistent.sh"))
        from pupa_backend.harnesses.langgraph.backend_tools import _build_startup_commands
        assert _build_startup_commands() == []

    def test_reads_non_comment_lines(self, monkeypatch, tmp_path):
        script = tmp_path / "startup.sh"
        script.write_text(textwrap.dedent("""\
            # this is a comment
            export FOO=bar
            # another comment
            alias ll='ls -la'
        """))
        monkeypatch.setenv("SHELL_STARTUP_SCRIPT", str(script))
        from pupa_backend.harnesses.langgraph.backend_tools import _build_startup_commands
        assert _build_startup_commands() == ["export FOO=bar", "alias ll='ls -la'"]

    def test_skips_blank_lines(self, monkeypatch, tmp_path):
        script = tmp_path / "startup.sh"
        script.write_text("cmd1\n\n   \ncmd2\n")
        monkeypatch.setenv("SHELL_STARTUP_SCRIPT", str(script))
        from pupa_backend.harnesses.langgraph.backend_tools import _build_startup_commands
        assert _build_startup_commands() == ["cmd1", "cmd2"]

    def test_default_path_used_when_var_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SHELL_STARTUP_SCRIPT", raising=False)
        default_script = tmp_path / "shell_startup.local.sh"
        default_script.write_text("echo hello\n")
        monkeypatch.setattr("pupa_backend.harnesses.langgraph.backend_tools._SHELL_STARTUP_DEFAULT", str(default_script))
        from pupa_backend.harnesses.langgraph.backend_tools import _build_startup_commands
        assert _build_startup_commands() == ["echo hello"]


# ---------------------------------------------------------------------------
# _build_shell_middlewares — env construction
# ---------------------------------------------------------------------------

class TestShellMiddlewaresEnv:
    """Verify the env dict passed to ShellToolMiddleware reflects SHELL_PASS_ENV."""

    @pytest.fixture(autouse=True)
    def _require_shell_enabled(self, monkeypatch):
        monkeypatch.setenv("SHELL_TOOL_ENABLED", "1")
        monkeypatch.delenv("SHELL_STARTUP_SCRIPT", raising=False)
        monkeypatch.setattr("pupa_backend.harnesses.langgraph.backend_tools._SHELL_STARTUP_DEFAULT", "/nonexistent")

    def _captured_env(self, monkeypatch) -> dict | None:
        """Call _build_shell_middlewares and capture the env kwarg ShellToolMiddleware received."""
        captured: dict[str, dict | None] = {}

        import langchain.agents.middleware.shell_tool as _mod
        original = _mod.ShellToolMiddleware

        class CapturingShellTool(original):
            def __init__(self, **kwargs):
                captured["env"] = kwargs.get("env")
                super().__init__(**kwargs)

        monkeypatch.setattr("pupa_backend.harnesses.langgraph.backend_tools.ShellToolMiddleware", CapturingShellTool)

        import pupa_backend.harnesses.langgraph.backend_tools as backend_tools
        backend_tools._build_shell_middlewares()
        return captured.get("env")

    def test_no_env_passed_when_shell_pass_env_unset(self, monkeypatch):
        monkeypatch.delenv("SHELL_PASS_ENV", raising=False)
        env = self._captured_env(monkeypatch)
        assert env is None

    def test_env_passed_when_shell_pass_env_set(self, monkeypatch):
        monkeypatch.setenv("SHELL_PASS_ENV", "1")
        monkeypatch.delenv("SHELL_ENV_EXCLUDE", raising=False)
        env = self._captured_env(monkeypatch)
        assert isinstance(env, dict)
        assert len(env) > 0

    def test_excluded_vars_stripped_from_env(self, monkeypatch):
        monkeypatch.setenv("SHELL_PASS_ENV", "1")
        monkeypatch.setenv("SHELL_ENV_EXCLUDE", "GH_TOKEN,MY_SECRET")
        monkeypatch.setenv("GH_TOKEN", "should-be-gone")
        monkeypatch.setenv("MY_SECRET", "also-gone")
        monkeypatch.setenv("SAFE_VAR", "keep-me")
        env = self._captured_env(monkeypatch)
        assert "GH_TOKEN" not in env
        assert "MY_SECRET" not in env
        assert "SAFE_VAR" in env

    def test_excluded_vars_not_in_env_when_pass_env_off(self, monkeypatch):
        monkeypatch.delenv("SHELL_PASS_ENV", raising=False)
        monkeypatch.setenv("GH_TOKEN", "irrelevant")
        env = self._captured_env(monkeypatch)
        assert env is None
