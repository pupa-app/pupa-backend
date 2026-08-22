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
        # No underscore before PASSWORD — a suffix glob misses this, and it's
        # the standard libpq variable.
        "PGPASSWORD",
        "MYSQL_PWD",
        # Connection strings with inline credentials.
        "REDIS_URL",
        "POSTGRES_URL",
        "MONGODB_URI",
        "AMQP_URL",
        # Bare `*_KEY`, no API in the name.
        "OPENAI_KEY",
        "NOTION_KEY",
        "SUPABASE_KEY",
        # Embeds the project key.
        "SENTRY_DSN",
        # Grant the *use* of a credential rather than being one.
        "SSH_AUTH_SOCK",
        "KUBECONFIG",
        "DOCKER_AUTH_CONFIG",
        # Case shouldn't matter.
        "gh_token",
        "aws_secret_access_key",
    ],
)
def test_secret_shaped_names_are_excluded_by_default(name: str) -> None:
    assert is_secret_env_name(name), name
    assert shell_env_excluded(name), name


@pytest.mark.parametrize(
    "name",
    ["PATH", "HOME", "USER", "LANG", "TERM", "PWD", "SHELL", "NVM_DIR",
     "TMPDIR", "LANGFUSE_BASE_URL", "SHELL_TOOL_WORKSPACE"],
)
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


# ---------------------------------------------------------------------------
# The claude_code sub-agent env
# ---------------------------------------------------------------------------


def test_api_billing_opt_in_still_forwards_the_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CLAUDE_CODE_ALLOW_API_BILLING=1` exists to pass the API credentials to
    the sub-agent. The generic secret denylist must not quietly override it —
    that would hand the subprocess `ANTHROPIC_BASE_URL` with no key to use
    against it, which fails in a confusing way rather than an obvious one."""
    from pupa_backend.harnesses.langgraph import claude_code_tool

    monkeypatch.setenv("CLAUDE_CODE_PASS_ENV", "1")
    monkeypatch.setenv("CLAUDE_CODE_ALLOW_API_BILLING", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-x")

    env = claude_code_tool._build_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-x"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "tok"
    assert env["AWS_ACCESS_KEY_ID"] == "AKIA-x"


def test_without_the_opt_in_credentials_are_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default: subscription-billed, so the API credentials must not reach the
    sub-agent — including `ANTHROPIC_BASE_URL`, which isn't secret-shaped and
    would otherwise slip past the denylist."""
    from pupa_backend.harnesses.langgraph import claude_code_tool

    monkeypatch.setenv("CLAUDE_CODE_PASS_ENV", "1")
    monkeypatch.delenv("CLAUDE_CODE_ALLOW_API_BILLING", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")

    env = claude_code_tool._build_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_BASE_URL" not in env


def test_other_secrets_are_stripped_even_with_api_billing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`allow_api` opts in to the *Claude* credentials, not to everything."""
    from pupa_backend.harnesses.langgraph import claude_code_tool

    monkeypatch.setenv("CLAUDE_CODE_PASS_ENV", "1")
    monkeypatch.setenv("CLAUDE_CODE_ALLOW_API_BILLING", "1")
    monkeypatch.setenv("PUPA_API_KEY", "operator-key")
    monkeypatch.setenv("GH_TOKEN", "ghp-x")

    env = claude_code_tool._build_env()
    assert "PUPA_API_KEY" not in env
    assert "GH_TOKEN" not in env
