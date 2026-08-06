"""pupa-backend setup wizard.

Run via `make setup` (or `uv run python scripts/setup.py`).

Creates ~/.pupa-backend/config.yml with the chosen configuration, optionally
generates a self-signed TLS certificate in ~/.pupa-backend/tls/, and
optionally installs a launchd (macOS) or systemd (Linux) service so
the backend starts automatically and restarts on failure.

After setup, run `make pair` to mint a QR pairing code for your iPhone.
"""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".pupa-backend"
YAML_FILE = CONFIG_DIR / "config.yml"
TLS_DIR = CONFIG_DIR / "tls"


# ---------------------------------------------------------------------------
# Colors & art
# ---------------------------------------------------------------------------

_R = "\033[91m"   # bright red
_C = "\033[36m"   # cyan — questions
_Y = "\033[33m"   # yellow — defaults / option numbers
_G = "\033[32m"   # green — success
_D = "\033[2m"    # dim — secondary text
_B = "\033[1m"    # bold
_X = "\033[0m"    # reset


PUPA_ASCII = r"""  
 ____                    
|  _ \ _   _ _ __   __ _ 
| |_) | | | | '_ \ / _` |
|  __/| |_| | |_) | (_| |
|_|    \__,_| .__/ \__,_|
            |_|          
 """
_CHAMELEON = _R + f"\n  {PUPA_ASCII}\n" + _X


def _print_welcome() -> None:
    print(_CHAMELEON)
    print(f"  {_B}{_R}Pupa{_X}  backend setup wizard")
    print()
    print(f"  {_D}Writes ~/.pupa-backend/config.yml  ·  loaded at every startup{_X}")
    print()
    print(f"  {_D}{'─' * 44}{_X}")
    print()


def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    suffix = f"  {_Y}[{default}]{_X}" if default else ""
    full = f"  {_C}{prompt}{_X}{suffix}: "
    if secret:
        import getpass
        val = getpass.getpass(full)
    else:
        val = input(full)
    return val.strip() or default


def _choose(prompt: str, options: list[tuple[str, str]], default: str) -> str:
    """Display a numbered menu and return the chosen key."""
    print(f"  {_C}{prompt}{_X}")
    for i, (key, label) in enumerate(options, 1):
        marker = f"  {_Y}← default{_X}" if key == default else ""
        print(f"    {_Y}{i}{_X}) {label}{marker}")
    while True:
        raw = input(f"  {_C}Choose{_X} [1-{len(options)}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print(f"  {_R}Invalid choice — try again.{_X}")


def _yesno(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {_C}{prompt}{_X} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _provider_desc(cfg: dict) -> str:
    """Compact one-line description of a provider config dict."""
    ptype = cfg.get("provider", "?")
    if ptype == "openrouter":
        suffix = " (key stored)" if cfg.get("api_key") else " (key from shell)"
        return f"openrouter  model={cfg.get('model', '')}{suffix}"
    if ptype == "openai_compatible":
        return f"openai_compatible  {cfg.get('base_url', '')}  model={cfg.get('model', '')}"
    if ptype == "bedrock":
        return f"bedrock  aws_profile={cfg.get('aws_profile', 'default')}"
    if ptype == "anthropic":
        suffix = " (key stored)" if cfg.get("api_key") else " (key from shell)"
        return f"anthropic{suffix}"
    return ptype


def _add_provider_interactively(existing_names: list[str]) -> tuple[str, dict]:
    """Prompt for a new LLM provider config; return (name, cfg_dict)."""
    ptype = _choose(
        "Provider type:",
        [
            ("openrouter",        "OpenRouter (native — key from OPENROUTER_API_KEY in shell)"),
            ("openai_compatible", "OpenAI-compatible endpoint (custom proxy, Ollama, LM Studio, …)"),
            ("anthropic",         "Anthropic API — direct"),
            ("bedrock",           "AWS Bedrock"),
        ],
        default="openrouter",
    )
    print()

    name_default = {"openrouter": "openrouter", "openai_compatible": "openai",
                    "anthropic": "anthropic", "bedrock": "bedrock"}.get(ptype, ptype)
    if name_default in existing_names:
        name_default = f"{name_default}_2"

    while True:
        name = _ask("Config name (no spaces — used as the YAML key)", default=name_default)
        if not name:
            print(f"  {_R}Name cannot be empty.{_X}")
        elif " " in name:
            print(f"  {_R}No spaces allowed.{_X}")
        else:
            break

    cfg: dict = {"provider": ptype}
    print()

    if ptype == "openrouter":
        cfg["model"] = _ask("Model ID (OpenRouter slug)", default="anthropic/claude-sonnet-4.6")
        print(f"  {_D}API key read from OPENROUTER_API_KEY in your shell. Leave blank to keep it there.{_X}")
        key = _ask("OPENROUTER_API_KEY (blank → read from shell)", secret=True)
        if key:
            cfg["api_key"] = key
        else:
            print(f"  {_D}Key not stored — remember: export OPENROUTER_API_KEY=sk-or-...{_X}")
    elif ptype == "openai_compatible":
        cfg["base_url"] = _ask("Base URL", default="https://openrouter.ai/api/v1")
        cfg["api_key"]  = _ask("API key (blank → read LLM_API_KEY from shell)", secret=True)
        cfg["model"]    = _ask("Model ID", default="anthropic/claude-sonnet-4.6")
    elif ptype == "anthropic":
        print(f"  {_D}API key stored in config.yml (chmod 600). Leave blank to export in shell instead.{_X}")
        key = _ask("ANTHROPIC_API_KEY", secret=True)
        if key:
            cfg["api_key"] = key
        else:
            print(f"  {_D}Key not stored — remember: export ANTHROPIC_API_KEY=sk-ant-...{_X}")
    elif ptype == "bedrock":
        profile = _ask(
            "AWS_PROFILE (optional — blank uses 'default' profile from ~/.aws/)",
            default=os.environ.get("AWS_PROFILE", ""),
        )
        if profile:
            cfg["aws_profile"] = profile
        print(f"  {_D}AWS credentials are never stored in config.yml — use ~/.aws/ or shell env.{_X}")

    return name, cfg


def _tailscale_ip() -> str | None:
    """Return the Tailscale IPv4 address, or None if Tailscale is not running."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=3,
        )
        ip = result.stdout.strip()
        if result.returncode == 0 and ip:
            return ip
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _tailscale_hostname() -> str | None:
    """Return the MagicDNS hostname (e.g. mymac.tail12345.ts.net), or None."""
    try:
        import json
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            dns = data.get("Self", {}).get("DNSName", "").rstrip(".")
            if dns:
                return dns
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


# ---------------------------------------------------------------------------
# Cloudflare named tunnel (custom domain)
# ---------------------------------------------------------------------------

CLOUDFLARED_DIR = Path.home() / ".cloudflared"

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _cloudflared_config_yaml(tunnel_id: str, creds_file: str, hostname: str) -> str:
    """Build ~/.cloudflared/config.yml for a named tunnel routing <hostname> to
    the local backend. Pure (no I/O) so it stays unit-testable."""
    return (
        f"tunnel: {tunnel_id}\n"
        f"credentials-file: {creds_file}\n"
        "\n"
        "ingress:\n"
        f"  - hostname: {hostname}\n"
        "    service: http://localhost:8004\n"
        "  - service: http_status:404\n"
    )


def _cloudflared_logged_in() -> bool:
    """True once `cloudflared tunnel login` has written the account cert."""
    return (CLOUDFLARED_DIR / "cert.pem").exists()


def _find_tunnel_id(name: str) -> str | None:
    """Return the UUID of an existing (non-deleted) named tunnel, or None."""
    try:
        proc = subprocess.run(
            ["cloudflared", "tunnel", "list", "--output", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return None
        for t in json.loads(proc.stdout or "[]"):
            # A live tunnel reports deleted_at as the Go zero-time
            # ("0001-01-01T00:00:00Z"), not null/missing — treat that as not deleted.
            deleted = str(t.get("deleted_at") or "")
            is_deleted = bool(deleted) and not deleted.startswith("0001-01-01")
            if t.get("name") == name and not is_deleted:
                return t.get("id")
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
        return None
    return None


def _create_tunnel(name: str) -> str | None:
    """Create a named tunnel; return its UUID parsed from cloudflared output."""
    proc = subprocess.run(
        ["cloudflared", "tunnel", "create", name],
        capture_output=True, text=True, timeout=30,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = _UUID_RE.search(out)
    return m.group(0) if (proc.returncode == 0 and m) else None


def _route_tunnel_dns(name: str, hostname: str) -> bool:
    """Point <hostname> at the named tunnel (creates the CNAME in Cloudflare DNS).
    An already-existing route counts as success."""
    proc = subprocess.run(
        ["cloudflared", "tunnel", "route", "dns", name, hostname],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode == 0:
        return True
    blob = ((proc.stdout or "") + (proc.stderr or "")).lower()
    return "already" in blob


def _configure_cloudflared_named(default_host: str, default_name: str) -> dict | None:
    """Full-auto named-tunnel setup: prompt for hostname + tunnel name, create the
    tunnel (if missing), route DNS, and write ~/.cloudflared/config.yml. Returns
    the pupa `cloudflared` block ({"hostname", "tunnel"}) on success, or None to
    fall back to the quick tunnel."""
    if not shutil.which("cloudflared"):
        print(f"  {_R}[!] cloudflared not found — install: brew install cloudflared{_X}")
        print(f"  {_D}    Falling back to the quick tunnel.{_X}")
        return None

    hostname = _ask("Public hostname (e.g. api.yourdomain.com)", default=default_host)
    if not hostname:
        print(f"  {_R}[!] No hostname — falling back to the quick tunnel.{_X}")
        return None
    name = _ask("Tunnel name", default=default_name or "pupa-backend")

    if not _cloudflared_logged_in():
        print()
        print(f"  {_R}[!] Not logged in to Cloudflare yet.{_X}")
        print(f"  {_D}    Run this once (opens a browser to pick your domain):{_X}")
        print(f"  {_Y}      cloudflared tunnel login{_X}")
        print(f"  {_D}    Then re-run `make setup` to finish the named tunnel.{_X}")
        return None

    tunnel_id = _find_tunnel_id(name)
    if tunnel_id:
        print(f"  {_D}Tunnel {name!r} already exists ({tunnel_id[:8]}…).{_X}")
    else:
        print(f"  {_D}Creating tunnel {name!r}…{_X}", end=" ", flush=True)
        tunnel_id = _create_tunnel(name)
        if not tunnel_id:
            print(f"{_R}failed.{_X}")
            print(f"  {_D}    Create it manually: cloudflared tunnel create {name}{_X}")
            return None
        print(f"{_G}done ({tunnel_id[:8]}…).{_X}")

    print(f"  {_D}Routing {hostname} → tunnel…{_X}", end=" ", flush=True)
    if _route_tunnel_dns(name, hostname):
        print(f"{_G}done.{_X}")
    else:
        print(f"{_R}failed.{_X}")
        print(f"  {_D}    Route manually: cloudflared tunnel route dns {name} {hostname}{_X}")

    creds_file = str(CLOUDFLARED_DIR / f"{tunnel_id}.json")
    cf_config = CLOUDFLARED_DIR / "config.yml"
    CLOUDFLARED_DIR.mkdir(parents=True, exist_ok=True)
    cf_config.write_text(_cloudflared_config_yaml(tunnel_id, creds_file, hostname))
    print(f"  {_G}✓ Wrote {cf_config}{_X}")

    return {"hostname": hostname, "tunnel": name}


# Forbidden env vars + accepted auth methods kept in sync with the loop's
# fail-closed billing guard — source of truth is backend/claude_loop/env.py
# (FORBIDDEN_ENV_VARS and _SUBSCRIPTION_AUTH_METHODS). The loop refuses to start
# if any forbidden var is present (they would divert billing off the
# subscription), so we warn about them here before the operator commits.
_LOOP_FORBIDDEN_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
)
_LOOP_SUBSCRIPTION_AUTH_METHODS: frozenset[str] = frozenset(
    {"claude.ai", "claudeai", "oauth_token"}
)


def _check_claude_subscription() -> tuple[bool, str]:
    """Soft preflight for the claude_code loop: True if `claude` reports a
    first-party subscription login. Never raises — returns (ok, message)."""
    forbidden = [k for k in _LOOP_FORBIDDEN_ENV_VARS if os.environ.get(k)]
    if forbidden:
        return False, (
            "these env var(s) will block the loop at startup: "
            f"{', '.join(forbidden)}"
        )
    binary = shutil.which("claude") or "claude"
    try:
        proc = subprocess.run(
            [binary, "auth", "status", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads((proc.stdout or "").strip())
        if not isinstance(data, dict):
            raise ValueError("unexpected shape")
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, TypeError):
        return False, "could not run `claude auth status` (CLI missing or not logged in)"
    ok = (
        bool(data.get("loggedIn"))
        and str(data.get("apiProvider")) == "firstParty"
        and str(data.get("authMethod")) in _LOOP_SUBSCRIPTION_AUTH_METHODS
    )
    if ok:
        return True, "subscription login confirmed"
    return False, (
        f"not a subscription login (loggedIn={data.get('loggedIn')}, "
        f"authMethod={data.get('authMethod')!r})"
    )


def _generate_tls_cert(hostname: str) -> tuple[Path, Path, str]:
    """Generate a self-signed CA + server cert in ~/.pupa-backend/tls/.

    Returns (cert_path, key_path, sha256_fingerprint_hex).
    Requires `cryptography` package (installed via `uv pip install -e '.[setup]'`).
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        import datetime
    except ImportError:
        print()
        print("  [!] 'cryptography' package not found.")
        print("      Install setup extras first:")
        print("      cd backend && uv pip install -e '.[setup]'")
        sys.exit(1)

    TLS_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = TLS_DIR / "server.crt"
    key_path = TLS_DIR / "server.key"

    # Generate private key.
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # SANs: localhost + the chosen Tailscale hostname or IP.
    san_dns = [x509.DNSName("localhost"), x509.DNSName(hostname)]
    san_ip: list[x509.IPAddress] = [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        san_ip.append(x509.IPAddress(ipaddress.ip_address(hostname)))
    except ValueError:
        pass  # hostname is not an IP; DNS SAN covers it

    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "pupa-backend"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Pupa"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(san_dns + san_ip),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)

    # Compute SHA-256 fingerprint of the DER-encoded cert (for QR / iOS pinning).
    import hashlib
    der = cert.public_bytes(serialization.Encoding.DER)
    fingerprint = hashlib.sha256(der).hexdigest()
    return cert_path, key_path, fingerprint


def _write_yaml(config: dict) -> None:
    """Write config dict to ~/.pupa-backend/config.yml."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        print()
        print("  [!] 'pyyaml' package not found.")
        print("      Install setup extras first:")
        print("      cd backend && uv pip install -e '.[setup]'")
        sys.exit(1)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        "# Pupa backend config — generated by `make setup`\n"
        "# Edit freely; re-run `make setup` to validate and regenerate.\n"
        "# openai_compatible and anthropic keys may be stored here (chmod 600).\n"
        "# AWS credentials are never stored — use ~/.aws/ or export env vars.\n\n"
    )
    YAML_FILE.write_text(header + yaml.dump(config, default_flow_style=False, sort_keys=False))
    YAML_FILE.chmod(0o600)


def _read_existing() -> dict[str, str]:
    """Return existing config as a flat {ENV_VAR: value} dict for wizard defaults."""
    if not YAML_FILE.exists():
        return {}
    try:
        import yaml  # type: ignore[import]
        from pupa_backend.pupa_config import _yaml_to_env_dict  # type: ignore[import]
        data = yaml.safe_load(YAML_FILE.read_text()) or {}
        return _yaml_to_env_dict(data)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def main() -> None:
    _print_welcome()

    existing = _read_existing()
    if existing:
        print(f"  {_D}Existing config found at {YAML_FILE} — values shown as defaults.{_X}")
        print()

    # Also load raw YAML so we can read the multi-provider block.
    try:
        import yaml as _yaml_mod
        existing_yaml: dict = _yaml_mod.safe_load(YAML_FILE.read_text()) or {} if YAML_FILE.exists() else {}
    except Exception:
        existing_yaml = {}

    # config dict that will be written as YAML
    config: dict = {}

    # ---- Backend agent harnesses ----
    # Several harnesses run at once, each mounted at POST /harnesses/{id} (the
    # default one also at POST /); the app picks which to talk to per backend
    # connection. claude_code is subscription-only and fail-closed, but the
    # credential stash lets it coexist with langgraph (which needs API keys).
    existing_harnesses: dict = existing_yaml.get("harnesses") or {}

    def _was_enabled(harness_id: str, default_on: bool) -> bool:
        cfg = existing_harnesses.get(harness_id)
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", False))
        # Migrate the retired single `agent_loop:` switch.
        legacy = existing.get("PUPA_AGENT_LOOP")
        if legacy:
            return legacy == harness_id
        return default_on

    deepagents_enabled = _yesno(
        "Enable the Deep Agents harness (LLM provider / API key)?",
        default=_was_enabled("deepagents", True),
    )
    claude_enabled = _yesno(
        "Enable the Claude Code harness (subscription-only, Claude Pro/Max)?",
        default=_was_enabled("claude_code", False),
    )
    if not (deepagents_enabled or claude_enabled):
        print(f"  {_D}At least one harness is required — enabling Deep Agents.{_X}")
        deepagents_enabled = True
    print()

    enabled_ids = [
        hid for hid, on in (("deepagents", deepagents_enabled), ("claude_code", claude_enabled)) if on
    ]
    if len(enabled_ids) == 1:
        default_harness = enabled_ids[0]
    else:
        default_harness = _choose(
            "Default harness (served at POST / for un-migrated clients):",
            [("deepagents", "Deep Agents"), ("claude_code", "Claude Code")],
            default=enabled_ids[0],
        )
        print()

    # ---- LLM providers ----
    # Collected only when the deepagents harness is enabled — the Claude Code
    # harness authenticates via the `claude` CLI subscription and ignores
    # LLM-provider config.
    providers_config: dict = {}
    default_provider: str = ""
    if deepagents_enabled:
        existing_providers: dict = existing_yaml.get("llm_providers") or {}
        existing_default: str = str(existing_yaml.get("default_llm_provider") or "")
        providers_config = dict(existing_providers)

        if providers_config:
            print(f"  {_D}Existing LLM provider configs:{_X}")
            for pname, pcfg in providers_config.items():
                tag = f"  {_Y}← default{_X}" if pname == existing_default else ""
                print(f"    {_C}{pname}{_X}  {_D}{_provider_desc(pcfg)}{_X}{tag}")
            print()
            add_more = _yesno("Add another provider config?", default=False)
        else:
            print(f"  {_D}No LLM providers configured yet — add at least one.{_X}")
            print()
            add_more = True

        while add_more:
            pname, pcfg = _add_provider_interactively(list(providers_config.keys()))
            providers_config[pname] = pcfg
            print()
            add_more = _yesno("Add another provider config?", default=False)
            print()

        pnames = list(providers_config.keys())
        if len(pnames) == 1:
            default_provider = pnames[0]
        else:
            default_provider = _choose(
                "Default LLM provider:",
                [(n, _provider_desc(providers_config[n])) for n in pnames],
                default=existing_default if existing_default in providers_config else pnames[0],
            )
        print()

        config["default_llm_provider"] = default_provider
        config["llm_providers"] = providers_config

    if claude_enabled:
        print(f"  {_D}Claude Code harness — authenticates via the `claude` CLI subscription.")
        print(f"  The credential stash keeps its `claude` subprocess off API-key billing.{_X}")
        print()
        ok, msg = _check_claude_subscription()
        if ok:
            print(f"  {_G}✓ {msg}.{_X}")
        else:
            print(f"  {_R}[!] {msg}.{_X}")
            print(f"  {_D}    Log in with `claude auth login` (Pro/Max) or set CLAUDE_CODE_OAUTH_TOKEN")
            print(f"  {_D}    from `claude setup-token` before starting.{_X}")
        print()

    # ---- Assemble the harnesses block ----
    harnesses_cfg: dict = {}
    if deepagents_enabled:
        harnesses_cfg["deepagents"] = {"enabled": True, "default": default_harness == "deepagents"}
    if claude_enabled:
        cc: dict = {"enabled": True, "default": default_harness == "claude_code"}
        # Preserve any existing Claude tuning (new nested spelling, else the
        # retired flat `claude_loop_*` / `claude_code_*` keys) so a re-run of the
        # wizard doesn't silently drop it.
        prev = existing_harnesses.get("claude_code")
        prev = prev if isinstance(prev, dict) else {}
        _flat_map = {
            "native": "claude_loop_native", "skills": "claude_loop_skills",
            "auto_approve": "claude_loop_auto_approve",
            "require_approval": "claude_loop_require_approval",
            "billing": "claude_loop_billing", "allow_api_billing": "claude_loop_allow_api_billing",
            "model": "claude_code_model", "workspace": "claude_code_workspace",
        }
        for nested_key, flat_key in _flat_map.items():
            if nested_key in prev:
                cc[nested_key] = prev[nested_key]
            elif flat_key in existing_yaml:
                cc[nested_key] = existing_yaml[flat_key]
        harnesses_cfg["claude_code"] = cc
    config["harnesses"] = harnesses_cfg

    # ---- Persistence ----
    # With no `DATABASE_URL` the backend defaults to persistent SQLite under
    # CONFIG_DIR (checkpoints.db + store.db, kept in separate files so
    # langgraph's two schemas don't collide), so there is nothing to write
    # here. Operators who want Postgres set `DATABASE_URL` in the environment.

    # ---- Connectivity ----
    print()
    existing_conn = existing.get("PUPA_CONNECTIVITY", "")
    if not existing_conn:
        existing_conn = "tailscale" if existing.get("PUPA_TLS_CERT") else "localhost"
    connectivity = _choose(
        "How will your iPhone reach the backend?",
        [
            ("tailscale", "Tailscale — stable private mesh (free, recommended for remote use)"),
            ("cloudflared", "Cloudflare tunnel — run `make tunnel` before pairing (URL changes on restart)"),
            ("localhost", "Localhost — same machine only"),
        ],
        default=existing_conn,
    )
    config["connectivity"] = connectivity

    fingerprint: str | None = None
    hostname: str = ""
    if connectivity == "tailscale":
        print()
        ts_ip = _tailscale_ip()
        ts_host = _tailscale_hostname()
        ts_default = ts_host or ts_ip or ""
        if ts_ip:
            print(f"  {_D}Detected Tailscale IP:   {_X}{ts_ip}")
        if ts_host:
            print(f"  {_D}Detected MagicDNS name:  {_X}{ts_host}")
        if not ts_ip:
            print(f"  {_R}[!] Tailscale not detected — install and start it first.{_X}")
            print(f"  {_D}    https://tailscale.com/download{_X}")
            print()
        hostname = _ask(
            "Hostname/IP to embed in certificate (MagicDNS name recommended)",
            default=existing.get("PUPA_HOSTNAME", ts_default),
        )
        print(f"  {_D}Generating self-signed certificate…{_X}", end=" ", flush=True)
        cert_path, key_path, fingerprint = _generate_tls_cert(hostname)
        print(f"{_G}done.{_X}")
        config["tls"] = {
            "cert": str(cert_path),
            "key": str(key_path),
            "hostname": hostname,
            "cert_fingerprint": fingerprint,
        }
    elif connectivity == "cloudflared":
        print()
        existing_cf = existing_yaml.get("cloudflared") or {}
        use_domain = _choose(
            "Do you have a domain on Cloudflare?",
            [("yes", "Yes — named tunnel on my domain (stable URL: pair once, never re-pair)"),
             ("no",  "No — quick tunnel (random trycloudflare.com URL, changes on restart)")],
            default="yes" if existing_cf else "no",
        )
        print()
        cf_block = None
        if use_domain == "yes":
            cf_block = _configure_cloudflared_named(
                default_host=str(existing_cf.get("hostname") or ""),
                default_name=str(existing_cf.get("tunnel") or "pupa-backend"),
            )
        if cf_block:
            config["cloudflared"] = cf_block
            hostname = cf_block["hostname"]
            print()
            print(f"  {_D}Named tunnel ready. `pupa-backend run` starts it for you")
            print(f"  (or run it standalone with `make tunnel-named`).{_X}")
            print()
        else:
            print(f"  {_D}Cloudflare quick tunnel — Cloudflare provides HTTPS, no certificate needed.")
            print(f"  Before pairing: run `make tunnel` in another terminal, then `make pair`.")
            print(f"  Note: the tunnel URL changes on every restart — you will need to re-pair.{_X}")
            print()
    else:  # localhost
        hostname = "localhost"
        print()
        print(f"  {_D}Localhost only — backend reachable on this machine only.{_X}")
        print()

    # ---- Auth ----
    print()
    print(f"  {_D}Auth: a PUPA_API_KEY lets you mint the first pairing code.")
    print(f"  After pairing your iPhone you can remove it.{_X}")
    existing_key = existing.get("PUPA_API_KEY", "")
    import secrets as _secrets
    default_key = existing_key or _secrets.token_hex(16)
    api_key = _ask("PUPA_API_KEY", default=default_key)
    if api_key:
        config["auth"] = {"api_key": api_key}

    # ---- Optional features ----
    print()
    if platform.system() == "Darwin":
        use_screenshare = _choose(
            "Enable screen-share (macOS-only — lets the agent see your screen via WebRTC):",
            [("yes", "Yes — mount the WebRTC signalling broker"),
             ("no", "No")],
            default="yes" if existing.get("PUPA_SCREENSHARE") else "no",
        )
        if use_screenshare == "yes":
            config["screenshare"] = True

    # Preserve any existing mcp_servers block (e.g. atlassian) the wizard itself
    # doesn't build, so re-running setup doesn't drop it. Playwright is now an
    # ordinary entry in this block, toggled by the prompt below.
    from pupa_backend.mcp_config_admin import (  # type: ignore[import]
        add_server,
        list_servers,
        load_config,
        playwright_entry,
        remove_server,
    )

    existing_servers = list_servers(load_config())
    if existing_servers:
        config["mcp_servers"] = dict(existing_servers)

    use_playwright = _choose(
        "Enable Playwright browser automation (requires Node.js + npx):",
        [("yes", "Yes — add @playwright/mcp to the agent"),
         ("no", "No")],
        default="yes" if "playwright" in existing_servers else "no",
    )
    if use_playwright == "yes":
        config = add_server(config, "playwright", playwright_entry(), overwrite=True)
        _install_playwright()
    elif "playwright" in config.get("mcp_servers", {}):
        config = remove_server(config, "playwright")

    # ---- Write config ----
    _write_yaml(config)
    print()
    print(f"  {_G}✓ Config written to {YAML_FILE}{_X}")

    # ---- Summary ----
    if connectivity == "cloudflared":
        cf = config.get("cloudflared")
        if cf:
            backend_url = f"https://{cf['hostname']}"
        else:
            backend_url = "https://<dynamic>.trycloudflare.com  (see `make tunnel`)"
    else:
        protocol = "https" if "tls" in config else "http"
        effective_hostname = config.get("tls", {}).get("hostname") or hostname or "localhost"
        backend_url = f"{protocol}://{effective_hostname}:8004"
    print()
    print(f"  {_R}┌───────────────────────────────────────────────────────┐{_X}")
    print(f"  {_R}│{_X}  {_B}Next steps{_X}                                           {_R}│{_X}")
    print(f"  {_R}├───────────────────────────────────────────────────────┤{_X}")
    print(f"  {_R}│{_X}  Backend URL: {_Y}{backend_url:<43}{_X}{_R}│{_X}")
    if fingerprint:
        print(f"  {_R}│{_X}  Cert SHA-256 (first 16): {_Y}{fingerprint[:16] + '…':<32}{_X}{_R}│{_X}")
    print(f"  {_R}├───────────────────────────────────────────────────────┤{_X}")
    print(f"  {_R}│{_X}  Start:   {_C}pupa-backend run{_X}                      {_R}│{_X}")
    print(f"  {_R}│{_X}           (or: make backend)                          {_R}│{_X}")
    print(f"  {_R}│{_X}  Pair:    {_C}pupa-backend pair{_X}                     {_R}│{_X}")
    print(f"  {_R}│{_X}  Service: {_C}pupa-backend service-install{_X}          {_R}│{_X}")
    if claude_enabled:
        print(f"  {_R}├───────────────────────────────────────────────────────┤{_X}")
        print(f"  {_R}│{_X}  Harness: {_Y}claude_code{_X} (subscription-only)             {_R}│{_X}")
        print(f"  {_R}│{_X}  Needs: {_C}claude auth login{_X} (Pro/Max)                  {_R}│{_X}")
    if deepagents_enabled:
        active_cfg = providers_config.get(default_provider, {})
        active_ptype = active_cfg.get("provider", "")
        if active_ptype == "openrouter" and not active_cfg.get("api_key"):
            print(f"  {_R}├───────────────────────────────────────────────────────┤{_X}")
            print(f"  {_R}│{_X}  OpenRouter key not stored — export in your shell:    {_R}│{_X}")
            print(f"  {_R}│{_X}    {_Y}export OPENROUTER_API_KEY=sk-or-...{_X}                 {_R}│{_X}")
        elif active_ptype == "anthropic" and not active_cfg.get("api_key"):
            print(f"  {_R}├───────────────────────────────────────────────────────┤{_X}")
            print(f"  {_R}│{_X}  Anthropic key not stored — export in your shell:     {_R}│{_X}")
            print(f"  {_R}│{_X}    {_Y}export ANTHROPIC_API_KEY=sk-ant-...{_X}                {_R}│{_X}")
        elif active_ptype == "bedrock" and not active_cfg.get("aws_profile"):
            print(f"  {_R}├───────────────────────────────────────────────────────┤{_X}")
            print(f"  {_R}│{_X}  AWS creds not stored — export in your shell:         {_R}│{_X}")
            print(f"  {_R}│{_X}    {_Y}export AWS_PROFILE=my-profile{_X}                       {_R}│{_X}")
    print(f"  {_R}└───────────────────────────────────────────────────────┘{_X}")
    print()

    # ---- Shell-env override warning ----
    # Shell env always wins over config.yml at load time (pupa_config). A stale
    # `export LLM_PROVIDER=...` in ~/.zshrc silently shadows the provider chosen
    # here — the backend then boots the wrong provider and crashes on missing
    # creds. Warn loudly if the current shell disagrees with the config default.
    if deepagents_enabled:
        active_cfg = providers_config.get(default_provider, {})
        active_ptype = active_cfg.get("provider", "")
        shell_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if shell_provider and active_ptype and shell_provider != active_ptype:
            print(f"  {_R}⚠ Shell override:{_X} your shell exports "
                  f"{_Y}LLM_PROVIDER={shell_provider}{_X}, but this config's default "
                  f"is {_Y}{active_ptype}{_X}.")
            print(f"    Shell env always wins over config.yml, so the backend will "
                  f"use {_Y}{shell_provider}{_X} and ignore your choice above.")
            print(f"    Fix: remove `export LLM_PROVIDER={shell_provider}` from your "
                  f"shell rc (e.g. ~/.zshrc), or run "
                  f"{_C}LLM_PROVIDER={active_ptype} pupa-backend run{_X}.")
            print()

    # ---- Optionally install service ----
    install_svc = _choose(
        "Install as background service (auto-start + restart on failure)?",
        [("yes", "Yes — install launchd/systemd service now"),
         ("no", "No — I'll start manually with `pupa-backend run`")],
        default="no",
    )
    if install_svc == "yes":
        _install_service()


def _install_playwright() -> None:
    backend_dir = Path(__file__).parent.parent
    if not (backend_dir / ".venv").exists():
        print(f"  {_R}[!] venv not found — run `make install-backend` first, then re-run setup.{_X}")
        return
    uv = shutil.which("uv")
    if not uv:
        print(f"  {_R}[!] uv not found — run: curl -LsSf https://astral.sh/uv/install.sh | sh{_X}")
        return
    print(f"  {_D}Syncing MCP deps (core) for Playwright…{_X}", end=" ", flush=True)
    result = subprocess.run(
        [uv, "pip", "install", "-e", "."],
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"{_R}failed.{_X}")
        print(result.stderr[-500:] if result.stderr else "")
        print(f"  {_D}Run manually: cd backend && uv pip install -e '.'{_X}")
        return
    print(f"{_G}done.{_X}")
    if not shutil.which("npx"):
        print(f"  {_R}[!] npx not found — install Node.js: brew install node{_X}")
        print(f"  {_D}    Then run: npx --yes playwright install chromium{_X}")
        return
    # Stream output (no capture) so the download shows progress, and so npx's
    # "Need to install playwright@…? (y)" prompt can't deadlock on a captured,
    # disconnected stdin — that hang made onboarding look frozen. `--yes` auto-
    # accepts that prompt; `chromium` is the only engine @playwright/mcp needs
    # by default (skips the firefox + webkit downloads).
    print(f"  {_D}Installing Playwright browser (chromium)…{_X}")
    result = subprocess.run(
        ["npx", "--yes", "playwright", "install", "chromium"],
        env={**os.environ, "NODE_TLS_REJECT_UNAUTHORIZED": "0"},
    )
    if result.returncode != 0:
        print(f"  {_R}[!] Playwright browser install failed.{_X}")
        print(f"  {_D}    Run manually: npx --yes playwright install chromium{_X}")
    else:
        print(f"  {_G}✓ Playwright browser ready.{_X}")


def _install_service() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "service", Path(__file__).parent / "service.py"
    )
    svc = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(svc)  # type: ignore[union-attr]
    backend_dir = Path(__file__).parent.parent
    svc.install(backend_dir=backend_dir)


if __name__ == "__main__":
    main()
