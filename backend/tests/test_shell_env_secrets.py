"""Secrets don't ride into the shell subprocess by default.

`SHELL_PASS_ENV=1` forwards the backend's environment to the shell tool — which
is a tool the *model* drives. Before this, the only thing standing between an
LLM-authored `env` and every provider key the operator exported was
`SHELL_ENV_EXCLUDE`, an operator-maintained list that is empty until someone
fills it in. The pass-through stays (startup scripts need PATH/HOME), but
secret-shaped names are now dropped unless explicitly allowed back.
"""


import pytest

from pupa_backend.harnesses.langgraph.backend_tools import (
    is_secret_env_name,
    shell_env_excluded,
)


@pytest.mark.parametrize(
    "name",
    [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "OPENROUTER_API_KEY",
        "PUPA_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LANGFUSE_SECRET_KEY",
        "DATABASE_URL",
        "SOME_VENDOR_PASSWORD",
        "MY_PRIVATE_KEY",
    ],
)
def test_secret_shaped_names_are_excluded_by_default(name: str) -> None:
    assert is_secret_env_name(name), name
    assert shell_env_excluded(name), name


@pytest.mark.parametrize("name", ["PATH", "HOME", "USER", "LANG", "TERM", "PWD", "SHELL"])
def test_ordinary_names_pass_through(name: str) -> None:
    """The startup script needs these — `gh` wrappers and anything else in
    `shell_startup.local.sh` break without PATH/HOME."""
    assert not is_secret_env_name(name), name
    assert not shell_env_excluded(name), name


def test_operator_exclusions_still_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL_ENV_EXCLUDE", "NVM_DIR")
    assert shell_env_excluded("NVM_DIR")


def test_allow_list_puts_a_named_secret_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escape hatch: a startup script that genuinely needs one credential (a
    `gh auth` wrapper, say) can name it rather than turning the default off."""
    monkeypatch.setenv("SHELL_ENV_ALLOW", "GH_TOKEN")
    assert is_secret_env_name("GH_TOKEN")
    assert not shell_env_excluded("GH_TOKEN")
    # Only the named one.
    assert shell_env_excluded("ANTHROPIC_API_KEY")


def test_allow_list_beats_the_operator_exclude_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL_ENV_EXCLUDE", "GH_TOKEN")
    monkeypatch.setenv("SHELL_ENV_ALLOW", "GH_TOKEN")
    assert not shell_env_excluded("GH_TOKEN")


def test_shell_middleware_env_drops_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the thing that actually builds the subprocess env."""
    from pupa_backend.harnesses.langgraph import backend_tools

    monkeypatch.setenv("SHELL_PASS_ENV", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("PUPA_API_KEY", "operator-key-should-not-leak")
    monkeypatch.setenv("HARMLESS_VAR", "fine")

    env = backend_tools._shell_subprocess_env()
    assert env is not None
    assert "ANTHROPIC_API_KEY" not in env
    assert "PUPA_API_KEY" not in env
    assert env.get("HARMLESS_VAR") == "fine"
    assert "PATH" in env


def test_shell_middleware_env_is_none_without_pass_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from pupa_backend.harnesses.langgraph import backend_tools

    monkeypatch.delenv("SHELL_PASS_ENV", raising=False)
    assert backend_tools._shell_subprocess_env() is None
