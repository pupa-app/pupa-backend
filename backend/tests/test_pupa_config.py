"""Tests for `pupa_config` — YAML → env-var mapping.

Covers:
  - ``default_llm_provider`` + ``llm_providers`` block for each provider type
  - Per-provider field → env-var mappings (bedrock, anthropic, openai_compatible)
  - Missing / empty blocks fall back gracefully
  - The real ``deploy/cloud-config.yml`` baked into the Railway image
"""

from pathlib import Path

import pytest
import yaml

from pupa_backend.pupa_config import _resolve_active_llm_provider, _yaml_to_env_dict


# ---------------------------------------------------------------------------
# _resolve_active_llm_provider — unit tests
# ---------------------------------------------------------------------------

def test_resolve_missing_block_returns_empty() -> None:
    assert _resolve_active_llm_provider({}) == {}


def test_resolve_missing_default_falls_back_to_only_entry() -> None:
    """No default_llm_provider → use the single configured entry."""
    data = {"llm_providers": {"anthropic": {"provider": "anthropic"}}}
    assert _resolve_active_llm_provider(data) == {"LLM_PROVIDER": "anthropic"}


def test_resolve_missing_default_falls_back_to_first_of_many() -> None:
    """No default_llm_provider → use the FIRST entry in YAML document order."""
    data = {
        "llm_providers": {
            "openrouter": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            "bedrock":    {"provider": "bedrock", "aws_profile": "work-sso"},
        },
    }
    result = _resolve_active_llm_provider(data)
    assert result["LLM_PROVIDER"] == "openrouter"
    assert result["LLM_MODEL"] == "anthropic/claude-sonnet-4.6"
    assert "AWS_PROFILE" not in result


def test_resolve_unknown_default_returns_empty() -> None:
    data = {
        "default_llm_provider": "ghost",
        "llm_providers": {"bedrock": {"provider": "bedrock"}},
    }
    assert _resolve_active_llm_provider(data) == {}


def test_resolve_bedrock_minimal() -> None:
    data = {
        "default_llm_provider": "bedrock",
        "llm_providers": {"bedrock": {"provider": "bedrock"}},
    }
    result = _resolve_active_llm_provider(data)
    assert result == {"LLM_PROVIDER": "bedrock"}


def test_resolve_bedrock_with_profile() -> None:
    data = {
        "default_llm_provider": "bedrock",
        "llm_providers": {"bedrock": {"provider": "bedrock", "aws_profile": "my-sso"}},
    }
    result = _resolve_active_llm_provider(data)
    assert result["LLM_PROVIDER"] == "bedrock"
    assert result["AWS_PROFILE"] == "my-sso"


def test_resolve_anthropic_without_key() -> None:
    data = {
        "default_llm_provider": "anthropic",
        "llm_providers": {"anthropic": {"provider": "anthropic"}},
    }
    result = _resolve_active_llm_provider(data)
    assert result == {"LLM_PROVIDER": "anthropic"}
    assert "ANTHROPIC_API_KEY" not in result


def test_resolve_anthropic_with_stored_key() -> None:
    data = {
        "default_llm_provider": "anthropic",
        "llm_providers": {"anthropic": {"provider": "anthropic", "api_key": "sk-ant-abc"}},
    }
    result = _resolve_active_llm_provider(data)
    assert result["LLM_PROVIDER"] == "anthropic"
    assert result["ANTHROPIC_API_KEY"] == "sk-ant-abc"


def test_resolve_openai_compatible_full() -> None:
    data = {
        "default_llm_provider": "openrouter",
        "llm_providers": {
            "openrouter": {
                "provider": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-or-test",
                "model": "anthropic/claude-sonnet-4.6",
            }
        },
    }
    result = _resolve_active_llm_provider(data)
    assert result["LLM_PROVIDER"] == "openai_compatible"
    assert result["LLM_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert result["LLM_API_KEY"] == "sk-or-test"
    assert result["LLM_MODEL"] == "anthropic/claude-sonnet-4.6"


def test_resolve_openai_compatible_partial_fields_omitted() -> None:
    """Fields that are absent or empty are not emitted."""
    data = {
        "default_llm_provider": "local",
        "llm_providers": {
            "local": {
                "provider": "openai_compatible",
                "base_url": "http://localhost:11434/v1",
                # api_key and model omitted
            }
        },
    }
    result = _resolve_active_llm_provider(data)
    assert result["LLM_BASE_URL"] == "http://localhost:11434/v1"
    assert "LLM_API_KEY" not in result
    assert "LLM_MODEL" not in result


def test_resolve_picks_named_default_not_first_entry() -> None:
    """default_llm_provider selects the right entry regardless of dict order."""
    data = {
        "default_llm_provider": "work",
        "llm_providers": {
            "personal": {"provider": "anthropic", "api_key": "sk-ant-personal"},
            "work": {"provider": "bedrock", "aws_profile": "work-sso"},
        },
    }
    result = _resolve_active_llm_provider(data)
    assert result["LLM_PROVIDER"] == "bedrock"
    assert result["AWS_PROFILE"] == "work-sso"
    assert "ANTHROPIC_API_KEY" not in result


# ---------------------------------------------------------------------------
# _yaml_to_env_dict integration
# ---------------------------------------------------------------------------

def test_non_llm_keys_still_mapped() -> None:
    """Other _YAML_TO_ENV entries are unaffected by the new provider logic."""
    data = {
        "default_llm_provider": "bedrock",
        "llm_providers": {"bedrock": {"provider": "bedrock"}},
        "screenshare": True,
        "auth": {"api_key": "secret"},
    }
    result = _yaml_to_env_dict(data)
    assert result["PUPA_SCREENSHARE"] == "1"
    assert result["PUPA_API_KEY"] == "secret"


def test_skills_disabled_opt_out_mapping() -> None:
    """`skills_disabled` is a negative-sense gate: true → PUPA_SKILLS_DISABLED=1,
    absent/false → unset (skills stay on)."""
    assert _yaml_to_env_dict({"skills_disabled": True})["PUPA_SKILLS_DISABLED"] == "1"
    assert "PUPA_SKILLS_DISABLED" not in _yaml_to_env_dict({"skills_disabled": False})
    assert "PUPA_SKILLS_DISABLED" not in _yaml_to_env_dict({})


# ---------------------------------------------------------------------------
# deploy/cloud-config.yml — regression test
# ---------------------------------------------------------------------------
# Pins the Railway image's baked-in config against schema drift. The previous
# version of this file used a flat ``llm_provider:`` key that the loader did
# not understand, so ``LLM_PROVIDER`` was silently unset and the FastAPI
# lifespan crashed before binding a port. Any future drift between
# ``cloud-config.yml`` and the YAML schema must fail this test.

CLOUD_CONFIG_PATH = Path(__file__).resolve().parents[2] / "deploy" / "cloud-config.yml"


def test_cloud_config_resolves_to_expected_env() -> None:
    data = yaml.safe_load(CLOUD_CONFIG_PATH.read_text())
    env = _yaml_to_env_dict(data)

    # Default provider resolves cleanly — no LLM_PROVIDER env var needed.
    assert env["LLM_PROVIDER"] == "anthropic"
    # No secrets baked into the YAML — they come from Railway env vars.
    assert "ANTHROPIC_API_KEY" not in env

    # Tracing is opt-out: on by default with no flag, and the cloud config
    # does not disable it. Secrets stay env-only.
    assert "PUPA_LANGFUSE_DISABLED" not in env
    assert "LANGFUSE_PUBLIC_KEY" not in env
    assert "LANGFUSE_SECRET_KEY" not in env

    # Safety gates: boolean ``false`` → env var omitted, which is the
    # "disabled" signal. An operator setting these on Railway would
    # override the safety posture — docs/deploy.md warns against this.
    assert "SHELL_TOOL_ENABLED" not in env
    assert "PUPA_SCREENSHARE" not in env

    # Skills are on by default but pinned off in cloud via the opt-out gate
    # (`skills_disabled: true` → PUPA_SKILLS_DISABLED=1).
    assert env["PUPA_SKILLS_DISABLED"] == "1"

    # Hard requirement: deploy must end up on Postgres. Startup fails if
    # DATABASE_URL is missing or resolves to a different backend — guards
    # against silent in-memory / SQLite fallback in a multi-tenant deploy.
    assert env["PUPA_REQUIRE_DB_SCHEME"] == "postgresql"


def test_cloud_config_lists_both_supported_providers() -> None:
    """Both providers must be present so operators can switch by setting
    ``LLM_PROVIDER`` on Railway without rebuilding the image."""
    data = yaml.safe_load(CLOUD_CONFIG_PATH.read_text())
    providers = data.get("llm_providers") or {}
    assert {"anthropic", "bedrock"}.issubset(providers.keys())
    assert providers["anthropic"]["provider"] == "anthropic"
    assert providers["bedrock"]["provider"] == "bedrock"


def test_cloud_config_yaml_matches_expected_structure() -> None:
    """Strict structural snapshot of ``deploy/cloud-config.yml``.

    The resolved-env tests above prove the *output* of the loader; this
    one pins the literal YAML so any drift in the source — adding a key,
    flipping a safety gate from ``false`` to ``true``, removing a
    provider entry — fails fast even when the env-var output happens to
    coincide (e.g. ``shell_tool_enabled: false`` and an omitted key both
    resolve to "no ``SHELL_TOOL_ENABLED`` env var").

    Update this expected dict deliberately when you change the deployment
    posture, alongside the YAML and CHANGELOG.
    """
    data = yaml.safe_load(CLOUD_CONFIG_PATH.read_text())
    expected = {
        "default_llm_provider": "anthropic",
        "llm_providers": {
            "anthropic": {"provider": "anthropic"},
            "bedrock":   {"provider": "bedrock"},
        },
        # Multi-tenant safety: explicit `false`, never relying on omission.
        "shell_tool_enabled": False,
        "screenshare":        False,
        # The claude_code tool spawns a host subprocess — pinned off in cloud.
        "claude_code_disabled": True,
        # Optional Claude Code agent loop: cloud stays on LangGraph and forbids
        # native host tools (subscription-billed + stateful = single-tenant only).
        "agent_loop":         "deepagents",
        "claude_loop_native": "off",
        # Skills are on by default but pinned off in cloud (opt-out gate).
        "skills_disabled":    True,
        # Cloud must land on Postgres; SQLite/in-memory data dies with the container.
        "persistence": {"require_db_scheme": "postgresql"},
        # Reachable deploy: refuse plaintext. Railway terminates TLS and
        # forwards X-Forwarded-Proto, so nothing legitimate is blocked.
        "transport": {"require_https": True, "trusted_proxy": True},
        # Tracing is opt-out: on automatically once Langfuse secrets are
        # supplied via Railway env vars, so no `langfuse` block is needed.
    }
    assert data == expected, (
        "deploy/cloud-config.yml drifted from the deployment posture pinned "
        "by this test. If the change is intentional, update `expected` and "
        "the CHANGELOG together."
    )


def test_multiple_providers_only_default_exported() -> None:
    """Only the default provider's vars reach the env dict."""
    data = {
        "default_llm_provider": "anthropic",
        "llm_providers": {
            "anthropic": {"provider": "anthropic", "api_key": "sk-ant-a"},
            "openrouter": {
                "provider": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-or-b",
                "model": "anthropic/claude-sonnet-4.6",
            },
        },
    }
    result = _yaml_to_env_dict(data)
    assert result["LLM_PROVIDER"] == "anthropic"
    assert result["ANTHROPIC_API_KEY"] == "sk-ant-a"
    assert "LLM_BASE_URL" not in result
    assert "LLM_API_KEY" not in result
