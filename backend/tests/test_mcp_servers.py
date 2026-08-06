"""Tests for the config-driven MCP servers feature.

Covers the three seams:
  - `pupa_config._resolve_mcp_servers` — YAML `mcp_servers:` block → PUPA_MCP_SERVERS.
  - `mcp_servers._load_mcp_config` — JSON parse, `enabled: false` filtering,
    `${VAR}` interpolation, transport defaulting.
  - `mcp_servers.mcp_servers_lifecycle` — yields None when unconfigured; loads +
    concatenates tools and skips a failing server when configured (MCP client mocked).
  - `backend_tools.mcp_server_specs` / `all_specs` — per-server iOS discovery specs.
"""

import json
import sys
import types

import pytest

import pupa_backend.mcp_servers as mcp_servers
from pupa_backend.harnesses.langgraph.backend_tools import BACKEND_TOOLS, all_specs, mcp_server_specs
from pupa_backend.pupa_config import _resolve_mcp_servers, _yaml_to_env_dict

_ATLASSIAN = {
    "atlassian": {
        "command": "uvx",
        "args": ["mcp-atlassian"],
        "env": {"CONFLUENCE_API_TOKEN": "${CONFLUENCE_API_TOKEN}"},
    }
}


# ---------------------------------------------------------------------------
# pupa_config._resolve_mcp_servers
# ---------------------------------------------------------------------------

def test_resolve_absent_block_returns_empty() -> None:
    assert _resolve_mcp_servers({}) == {}


def test_resolve_empty_block_returns_empty() -> None:
    assert _resolve_mcp_servers({"mcp_servers": {}}) == {}


def test_resolve_non_dict_block_returns_empty() -> None:
    assert _resolve_mcp_servers({"mcp_servers": ["nope"]}) == {}


def test_resolve_serialises_block_to_json_env() -> None:
    out = _resolve_mcp_servers({"mcp_servers": _ATLASSIAN})
    assert set(out) == {"PUPA_MCP_SERVERS"}
    assert json.loads(out["PUPA_MCP_SERVERS"]) == _ATLASSIAN


def test_yaml_to_env_dict_includes_mcp_servers() -> None:
    out = _yaml_to_env_dict({"mcp_servers": _ATLASSIAN})
    assert json.loads(out["PUPA_MCP_SERVERS"]) == _ATLASSIAN


# ---------------------------------------------------------------------------
# mcp_servers._load_mcp_config
# ---------------------------------------------------------------------------

def test_load_unset_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_MCP_SERVERS", raising=False)
    assert mcp_servers._load_mcp_config() == {}


def test_load_bad_json_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_MCP_SERVERS", "{not json")
    assert mcp_servers._load_mcp_config() == {}


def test_load_defaults_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(_ATLASSIAN))
    conns = mcp_servers._load_mcp_config()
    assert conns["atlassian"]["transport"] == "stdio"
    # `enabled`/meta keys absent, real connection keys preserved.
    assert conns["atlassian"]["command"] == "uvx"


def test_load_url_defaults_streamable_http(monkeypatch: pytest.MonkeyPatch) -> None:
    block = {"remote": {"url": "https://host/mcp"}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    conns = mcp_servers._load_mcp_config()
    assert conns["remote"]["transport"] == "streamable_http"


def test_load_filters_disabled_and_strips_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    block = {
        "on": {"command": "a"},
        "off": {"command": "b", "enabled": False},
    }
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    conns = mcp_servers._load_mcp_config()
    assert set(conns) == {"on"}
    assert "enabled" not in conns["on"]


def test_load_interpolates_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "secret-tok")
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(_ATLASSIAN))
    conns = mcp_servers._load_mcp_config()
    assert conns["atlassian"]["env"]["CONFLUENCE_API_TOKEN"] == "secret-tok"


def test_load_leaves_unset_placeholder_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONFLUENCE_API_TOKEN", raising=False)
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(_ATLASSIAN))
    conns = mcp_servers._load_mcp_config()
    assert conns["atlassian"]["env"]["CONFLUENCE_API_TOKEN"] == "${CONFLUENCE_API_TOKEN}"


def test_load_strips_description_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    block = {"atlassian": {"command": "uvx", "description": "Confluence"}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    conns = mcp_servers._load_mcp_config()
    # `description` is ours (meta), not part of the MultiServerMCPClient connection.
    assert "description" not in conns["atlassian"]
    assert conns["atlassian"]["command"] == "uvx"


# ---------------------------------------------------------------------------
# mcp_servers._server_descriptions
# ---------------------------------------------------------------------------

def test_server_descriptions_enabled_only_and_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    block = {
        "a": {"command": "x", "description": "  Server A  "},
        "b": {"command": "y"},  # no description → absent
        "off": {"command": "z", "description": "D", "enabled": False},  # disabled → absent
    }
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    assert mcp_servers._server_descriptions() == {"a": "Server A"}


# ---------------------------------------------------------------------------
# backend_tools.mcp_server_specs / all_specs
# ---------------------------------------------------------------------------

def test_specs_unset_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_MCP_SERVERS", raising=False)
    assert mcp_server_specs() == []


def test_specs_one_per_enabled_server(monkeypatch: pytest.MonkeyPatch) -> None:
    block = {"atlassian": {"command": "uvx"}, "off": {"command": "x", "enabled": False}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    specs = mcp_server_specs()
    assert [s.name for s in specs] == ["mcp_atlassian"]
    assert specs[0].enabled_by_env is True


def test_all_specs_is_static_plus_dynamic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps({"atlassian": {"command": "uvx"}}))
    names = [s.name for s in all_specs()]
    assert names == [s.name for s in BACKEND_TOOLS] + ["mcp_atlassian"]


def test_specs_use_server_description(monkeypatch: pytest.MonkeyPatch) -> None:
    block = {"atlassian": {"command": "uvx", "description": "Confluence search + write"}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    assert mcp_server_specs()[0].description == "Confluence search + write"


def test_specs_fallback_description(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps({"atlassian": {"command": "uvx"}}))
    desc = mcp_server_specs()[0].description
    assert "mcp_servers block" in desc and "get_tools" in desc


# ---------------------------------------------------------------------------
# mcp_servers.mcp_servers_lifecycle
# ---------------------------------------------------------------------------

async def test_lifecycle_yields_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_MCP_SERVERS", raising=False)
    async with mcp_servers.mcp_servers_lifecycle() as mcp:
        assert mcp is None


def _install_fake_adapters(monkeypatch: pytest.MonkeyPatch, *, bad: set[str]) -> None:
    """Inject fake `langchain_mcp_adapters` modules so the lazy import resolves.

    `load_mcp_tools` raises for any session whose server name is in `bad` (to
    exercise the skip-on-failure path) and otherwise returns one stub tool named
    `<server>_tool`.
    """

    class _FakeSession:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

    class _FakeClient:
        def __init__(self, connections: dict) -> None:
            self.connections = connections

        def session(self, name: str) -> "_FakeSession":
            return _FakeSession(name)

    async def _fake_load(session: "_FakeSession") -> list:
        if session.name in bad:
            raise RuntimeError(f"{session.name} boom")
        return [types.SimpleNamespace(name=f"{session.name}_tool")]

    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = _FakeClient
    tools_mod = types.ModuleType("langchain_mcp_adapters.tools")
    tools_mod.load_mcp_tools = _fake_load
    pkg = types.ModuleType("langchain_mcp_adapters")
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", pkg)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.tools", tools_mod)


async def test_lifecycle_loads_and_concatenates_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_adapters(monkeypatch, bad=set())
    block = {"a": {"command": "x"}, "b": {"command": "y"}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    async with mcp_servers.mcp_servers_lifecycle() as mcp:
        assert mcp is not None
        assert sorted(t.name for t in mcp.tools) == ["a_tool", "b_tool"]
        assert mcp.server_tool_names["a"] == frozenset({"a_tool"})


async def test_lifecycle_surfaces_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_adapters(monkeypatch, bad=set())
    block = {"a": {"command": "x", "description": "Server A"}, "b": {"command": "y"}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    async with mcp_servers.mcp_servers_lifecycle() as mcp:
        assert mcp.server_descriptions == {"a": "Server A"}


async def test_lifecycle_skips_failing_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_adapters(monkeypatch, bad={"b"})
    block = {"a": {"command": "x"}, "b": {"command": "y"}}
    monkeypatch.setenv("PUPA_MCP_SERVERS", json.dumps(block))
    async with mcp_servers.mcp_servers_lifecycle() as mcp:
        assert mcp is not None
        # The good server is still served; the bad one is skipped, not fatal.
        assert [t.name for t in mcp.tools] == ["a_tool"]
        assert "b" not in mcp.server_tool_names
