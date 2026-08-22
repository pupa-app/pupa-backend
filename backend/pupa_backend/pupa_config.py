"""Load ~/.pupa-backend/config.yml into os.environ.

Shared by app.py (startup), pair.py, service.py, and the pupa-backend
shell wrapper.

Priority (lowest → highest):
    config.yml  <  shell environment

Shell env always wins — AWS_PROFILE exported in a terminal overrides config.yml.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".pupa-backend"
YAML_FILE = CONFIG_DIR / "config.yml"

# YAML dotted key → env var name.
_YAML_TO_ENV: dict[str, str] = {
    "persistence.database_url":      "DATABASE_URL",
    "persistence.require_db_scheme": "PUPA_REQUIRE_DB_SCHEME",
    # bool: `transport.require_https: true` → "1". Refuse plaintext requests.
    "transport.require_https": "PUPA_REQUIRE_HTTPS",
    # bool: `transport.trusted_proxy: true` → "1". Believe `X-Forwarded-*`.
    "transport.trusted_proxy": "PUPA_TRUSTED_PROXY",
    "tls.cert":              "PUPA_TLS_CERT",
    "tls.key":               "PUPA_TLS_KEY",
    "tls.hostname":          "PUPA_HOSTNAME",
    "tls.cert_fingerprint":  "PUPA_CERT_FINGERPRINT",
    "auth.api_key":          "PUPA_API_KEY",
    "screenshare":           "PUPA_SCREENSHARE",
    "connectivity":          "PUPA_CONNECTIVITY",
    # Named Cloudflare tunnel (custom domain). Set only when the operator runs a
    # persistent named tunnel — hostname gives a stable public URL (no random
    # trycloudflare.com), tunnel is the cloudflared tunnel name used by
    # `make tunnel-named` and the optional cloudflared service. Absent for the
    # quick-tunnel path (connectivity=cloudflared with no domain).
    "cloudflared.hostname":  "PUPA_CLOUDFLARED_HOSTNAME",
    "cloudflared.tunnel":    "PUPA_CLOUDFLARED_TUNNEL",
    "shell_tool_enabled":    "SHELL_TOOL_ENABLED",
    # claude_code is ON by default; this is an opt-OUT gate (negative sense), so
    # config `claude_code_disabled: true` maps to PUPA_CLAUDE_CODE_DISABLED=1.
    # Cloud pins it off; local installs leave it unset (claude_code enabled).
    "claude_code_disabled":  "PUPA_CLAUDE_CODE_DISABLED",
    "claude_code_workspace": "CLAUDE_CODE_WORKSPACE",
    "claude_code_model":     "CLAUDE_CODE_MODEL",
    # Claude Code harness knobs. STRING values (not booleans) must be quoted in
    # YAML — `claude_loop_native: "off"`. An unquoted `off`/`no` parses to YAML
    # False and would be omitted; always quote the literal string.
    # `claude_loop_allow_api_billing` is the only boolean here (omit ⇒ off; the
    # api-billing path is not implemented anyway). These are legacy flat keys;
    # the preferred spelling now nests them under `harnesses.claude_code.*`
    # (mapped by `_resolve_harnesses`), but the flat keys still work.
    "claude_loop_native":            "PUPA_CLAUDE_LOOP_NATIVE",
    "claude_loop_skills":            "PUPA_CLAUDE_LOOP_SKILLS",
    # Run every permitted command without a chat prompt (flow over friction).
    # Positive bool: `claude_loop_auto_approve: true` → "1".
    "claude_loop_auto_approve":      "PUPA_CLAUDE_LOOP_AUTO_APPROVE",
    # Opt back INTO a per-command approval prompt (default is run-freely). Positive
    # bool: `claude_loop_require_approval: true` → "1".
    "claude_loop_require_approval":  "PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL",
    "claude_loop_billing":           "PUPA_CLAUDE_LOOP_BILLING",
    "claude_loop_allow_api_billing": "PUPA_CLAUDE_LOOP_ALLOW_API_BILLING",
    # Skills are ON by default; this is an opt-OUT gate (negative sense), so a
    # config `skills_disabled: true` maps to PUPA_SKILLS_DISABLED=1. Cloud pins
    # it off; local installs leave it unset (skills enabled).
    "skills_disabled":       "PUPA_SKILLS_DISABLED",
    "tavily_api_key":        "TAVILY_API_KEY",
    # Langfuse is ON by default whenever credentials are present; this is an
    # opt-OUT gate, so `langfuse.disabled: true` maps to PUPA_LANGFUSE_DISABLED=1.
    "langfuse.disabled":     "PUPA_LANGFUSE_DISABLED",
    "langfuse.public_key":   "LANGFUSE_PUBLIC_KEY",
    "langfuse.secret_key":   "LANGFUSE_SECRET_KEY",
}


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _resolve_active_llm_provider(data: dict) -> dict[str, str]:
    """Extract LLM env vars from the llm_providers / default_llm_provider block.

    The config file is the single source of truth for the default provider — the
    Makefile no longer injects one. ``default_llm_provider`` names the active
    entry; when it's omitted, the **first** entry in ``llm_providers`` (YAML
    document order) is used, so a single-provider config "just works" without
    restating its name. A ``default_llm_provider`` that names a missing entry is
    treated as a config error and yields ``{}`` (startup then fails loudly with a
    clear error from ``_build_default_from_env`` about ``LLM_PROVIDER`` being unset).

    Returns an empty dict if the ``llm_providers`` block is absent/empty.

    Provider → env-var mapping:
      bedrock           → LLM_PROVIDER, AWS_PROFILE (optional)
      anthropic         → LLM_PROVIDER, ANTHROPIC_API_KEY (optional)
      openai_compatible → LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
      openrouter        → LLM_PROVIDER, LLM_MODEL, OPENROUTER_API_KEY (optional)
    """
    providers: dict = data.get("llm_providers") or {}
    if not providers:
        return {}
    default: str = str(data.get("default_llm_provider") or "").strip()
    if default:
        cfg = providers.get(default)
    else:
        # No explicit default — fall back to the first configured entry.
        _, cfg = next(iter(providers.items()))
    if not isinstance(cfg, dict) or not cfg:
        return {}
    ptype = str(cfg.get("provider") or "").lower().strip()
    if not ptype:
        return {}

    result: dict[str, str] = {"LLM_PROVIDER": ptype}
    if ptype == "bedrock":
        if v := str(cfg.get("aws_profile") or "").strip():
            result["AWS_PROFILE"] = v
    elif ptype == "anthropic":
        if v := str(cfg.get("api_key") or "").strip():
            result["ANTHROPIC_API_KEY"] = v
    elif ptype == "openai_compatible":
        for cfg_key, env_key in (
            ("base_url", "LLM_BASE_URL"),
            ("api_key",  "LLM_API_KEY"),
            ("model",    "LLM_MODEL"),
        ):
            if v := str(cfg.get(cfg_key) or "").strip():
                result[env_key] = v
    elif ptype == "openrouter":
        # Native OpenRouter: base_url is fixed in agent.py. api_key is optional —
        # blank falls back to OPENROUTER_API_KEY from the shell (shell wins on apply).
        if v := str(cfg.get("api_key") or "").strip():
            result["OPENROUTER_API_KEY"] = v
        if v := str(cfg.get("model") or "").strip():
            result["LLM_MODEL"] = v
    return result


def _resolve_mcp_servers(data: dict) -> dict[str, str]:
    """Serialise the structured `mcp_servers:` block to the PUPA_MCP_SERVERS env var.

    Like `llm_providers`, this is a nested block that can't ride the flat dotted
    `_YAML_TO_ENV` map, so it gets its own resolver. The whole block is passed
    through as JSON; `mcp_servers.py` parses it (filtering `enabled: false`,
    expanding `${VAR}`) at startup. Returns `{}` when the block is absent/empty.
    """
    block = data.get("mcp_servers")
    if not isinstance(block, dict) or not block:
        return {}
    return {"PUPA_MCP_SERVERS": json.dumps(block)}


# Per-harness config keys that map onto existing flat `PUPA_CLAUDE_LOOP_*` env
# vars, so `harnesses.claude_code.<key>` and the legacy flat key both work and
# gate.py/env.py env reads stay untouched.
_CLAUDE_HARNESS_KEY_TO_ENV: dict[str, str] = {
    "native":            "PUPA_CLAUDE_LOOP_NATIVE",
    "skills":            "PUPA_CLAUDE_LOOP_SKILLS",
    "auto_approve":      "PUPA_CLAUDE_LOOP_AUTO_APPROVE",
    "require_approval":  "PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL",
    "billing":           "PUPA_CLAUDE_LOOP_BILLING",
    "allow_api_billing": "PUPA_CLAUDE_LOOP_ALLOW_API_BILLING",
    "model":             "CLAUDE_CODE_MODEL",
    "workspace":         "CLAUDE_CODE_WORKSPACE",
    "config_dir":        "PUPA_CLAUDE_LOOP_CONFIG_DIR",
}


def _resolve_harnesses(data: dict) -> dict[str, str]:
    """Serialise the `harnesses:` block to `PUPA_HARNESSES` (JSON) + per-harness env.

    The `harnesses:` block replaces the single `agent_loop:` switch. Each entry is
    `{enabled: bool, default: bool, ...per-harness knobs}`. The block itself is
    passed through as JSON for `harnesses.build_registry()`; the Claude harness's
    nested knobs are additionally flattened onto the existing `PUPA_CLAUDE_LOOP_*`
    env vars so the loop's env reads don't change. Returns `{}` when absent.
    """
    block = data.get("harnesses")
    if not isinstance(block, dict) or not block:
        return {}
    result: dict[str, str] = {"PUPA_HARNESSES": json.dumps(block)}
    claude_cfg = block.get("claude_code")
    if isinstance(claude_cfg, dict):
        for cfg_key, env_var in _CLAUDE_HARNESS_KEY_TO_ENV.items():
            if cfg_key not in claude_cfg:
                continue
            v = claude_cfg[cfg_key]
            if isinstance(v, bool):
                if v:
                    result[env_var] = "1"
            elif str(v).strip():
                result[env_var] = str(v)
    return result


def _yaml_to_env_dict(data: dict) -> dict[str, str]:
    flat = _flatten(data)
    result: dict[str, str] = {}
    for yaml_key, env_var in _YAML_TO_ENV.items():
        v = flat.get(yaml_key)
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                result[env_var] = "1"
            # False → omit; unset var is the "disabled" signal
        elif str(v).strip():
            result[env_var] = str(v)
    result.update(_resolve_active_llm_provider(data))
    result.update(_resolve_mcp_servers(data))
    result.update(_resolve_harnesses(data))
    return result


def load_pupa_config(apply: bool = True) -> dict[str, str]:
    """Read config.yml and return {ENV_VAR: value}.

    When apply=True (default) each var is written to os.environ only if the
    shell hasn't already set it (shell env always wins).
    Pass apply=False to get the dict without touching os.environ — used by
    service.py when building the launchd/systemd env block.
    """
    import yaml  # type: ignore[import]

    if not YAML_FILE.exists():
        return {}

    data = yaml.safe_load(YAML_FILE.read_text()) or {}
    env_dict = _yaml_to_env_dict(data)

    if apply:
        for k, v in env_dict.items():
            if k not in os.environ:
                os.environ[k] = v

    return env_dict


def yaml_to_shell_exports() -> None:
    """Print `export KEY='VALUE'` lines for every config value.

    Called by the pupa-backend shell wrapper to pull config.yml
    into the shell environment before delegating to make targets.
    """
    env_dict = load_pupa_config(apply=False)
    for k, v in env_dict.items():
        safe = v.replace("'", "'\\''")
        print(f"export {k}='{safe}'")
