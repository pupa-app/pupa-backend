"""Tests for `mcp_config_admin` — the core behind the `pupa-backend mcp` CLI.

Covers entry construction, the add/remove/list mutations, name validation, and a
full round-trip: written config.yml → `pupa_config` → `PUPA_MCP_SERVERS` →
`mcp_servers._load_mcp_config`, proving the CLI's output is what the runtime
actually consumes.
"""

import json

import pytest
import yaml

import pupa_backend.mcp_servers as mcp_servers
from pupa_backend.mcp_config_admin import (
    add_server,
    atlassian_entry,
    build_entry,
    list_servers,
    load_config,
    playwright_entry,
    remove_server,
    validate_name,
    write_config,
    _pairs_to_dict,
)
from pupa_backend.pupa_config import _yaml_to_env_dict


# ── build_entry ─────────────────────────────────────────────────────────────

def test_build_entry_stdio() -> None:
    e = build_entry(command="npx", args=["-y", "srv"], env={"K": "v"})
    assert e == {"command": "npx", "args": ["-y", "srv"], "env": {"K": "v"}}


def test_build_entry_http() -> None:
    e = build_entry(url="https://h/mcp", transport="streamable_http", headers={"A": "b"})
    assert e == {"url": "https://h/mcp", "transport": "streamable_http", "headers": {"A": "b"}}


def test_build_entry_requires_exactly_one_target() -> None:
    with pytest.raises(ValueError):
        build_entry()
    with pytest.raises(ValueError):
        build_entry(command="x", url="https://h")


def test_build_entry_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError):
        build_entry(command="x", transport="carrier-pigeon")


def test_build_entry_omits_empty_args() -> None:
    assert build_entry(command="x", args=[]) == {"command": "x"}


def test_build_entry_with_description() -> None:
    e = build_entry(command="x", description="  browser automation  ")
    assert e["description"] == "browser automation"  # trimmed


def test_build_entry_omits_blank_description() -> None:
    assert "description" not in build_entry(command="x", description="   ")
    assert "description" not in build_entry(command="x")


def test_atlassian_entry_uses_token_placeholder() -> None:
    e = atlassian_entry(url="https://you.atlassian.net/wiki", username="me@corp.com")
    assert e["command"] == "uvx"
    assert e["args"] == ["mcp-atlassian"]
    assert e["env"]["CONFLUENCE_API_TOKEN"] == "${CONFLUENCE_API_TOKEN}"


def test_playwright_entry_shape() -> None:
    e = playwright_entry()
    assert e["command"] == "npx"
    assert e["args"] == ["@playwright/mcp@latest"]
    assert e["description"]  # ships a non-empty default description


# ── validate_name / _pairs_to_dict ──────────────────────────────────────────

@pytest.mark.parametrize("name", ["atlassian", "fs_1", "remote-api"])
def test_validate_name_ok(name: str) -> None:
    validate_name(name)


@pytest.mark.parametrize("name", ["", "has space", "bad/slash", "quote\"d"])
def test_validate_name_rejects(name: str) -> None:
    with pytest.raises(ValueError):
        validate_name(name)


def test_pairs_to_dict() -> None:
    assert _pairs_to_dict(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(ValueError):
        _pairs_to_dict(["noequals"])


# ── add_server / remove_server / list_servers ───────────────────────────────

def test_add_server_new() -> None:
    cfg = add_server({"auth": {"api_key": "k"}}, "fs", {"command": "npx"})
    assert cfg["mcp_servers"]["fs"] == {"command": "npx"}
    assert cfg["auth"] == {"api_key": "k"}  # unrelated keys preserved


def test_add_server_duplicate_without_overwrite_raises() -> None:
    cfg = {"mcp_servers": {"fs": {"command": "a"}}}
    with pytest.raises(ValueError):
        add_server(cfg, "fs", {"command": "b"})


def test_add_server_overwrite() -> None:
    cfg = {"mcp_servers": {"fs": {"command": "a"}}}
    out = add_server(cfg, "fs", {"command": "b"}, overwrite=True)
    assert out["mcp_servers"]["fs"] == {"command": "b"}


def test_remove_server_drops_empty_block() -> None:
    cfg = {"mcp_servers": {"fs": {"command": "a"}}}
    out = remove_server(cfg, "fs")
    assert "mcp_servers" not in out


def test_remove_server_missing_raises() -> None:
    with pytest.raises(KeyError):
        remove_server({"mcp_servers": {}}, "ghost")


def test_list_servers_handles_absent_block() -> None:
    assert list_servers({}) == {}


# ── file round-trip + runtime consumption ───────────────────────────────────

def test_write_then_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yml"
    cfg = add_server({}, "atlassian", atlassian_entry(url="https://h/wiki", username="me"))
    write_config(cfg, path=path)

    # Header preserved, perms locked down.
    assert path.read_text().startswith("# Pupa backend config")
    assert (path.stat().st_mode & 0o777) == 0o600

    assert load_config(path=path)["mcp_servers"]["atlassian"]["command"] == "uvx"


def test_cli_output_is_consumed_by_runtime(tmp_path, monkeypatch) -> None:
    """config.yml the CLI writes → pupa_config → PUPA_MCP_SERVERS → mcp_servers."""
    path = tmp_path / "config.yml"
    cfg = add_server({}, "fs", build_entry(command="npx", args=["-y", "srv"]))
    write_config(cfg, path=path)

    data = yaml.safe_load(path.read_text())
    env = _yaml_to_env_dict(data)
    assert "PUPA_MCP_SERVERS" in env

    monkeypatch.setenv("PUPA_MCP_SERVERS", env["PUPA_MCP_SERVERS"])
    conns = mcp_servers._load_mcp_config()
    assert conns["fs"]["command"] == "npx"
    assert conns["fs"]["transport"] == "stdio"  # defaulted by the loader
