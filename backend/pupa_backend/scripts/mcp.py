"""Operator CLI: manage MCP servers in ~/.pupa-backend/config.yml.

Run via `make mcp ARGS="..."` or the installed CLI `pupa-backend mcp ...`.

Subcommands::

    pupa-backend mcp list
    pupa-backend mcp add                         # interactive wizard
    pupa-backend mcp add --name atlassian \
        --confluence-url https://you.atlassian.net/wiki \
        --confluence-user you@corp.com           # Atlassian preset, non-interactive
    pupa-backend mcp add --playwright            # Playwright preset (browser automation)
    pupa-backend mcp add --name fs --command npx \
        --arg -y --arg @modelcontextprotocol/server-filesystem --arg /tmp
    pupa-backend mcp add --name remote --url https://host/mcp --transport streamable_http \
        --header "Authorization=Bearer ${SOME_TOKEN}"
    pupa-backend mcp remove atlassian

The mutation logic lives in `mcp_config_admin` (importable + unit-tested); this
file is the thin argparse + interactive-prompt shell. Editing config.yml does
NOT hot-reload a running backend — restart to apply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pupa_backend.mcp_config_admin import (  # type: ignore[import]
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
from pupa_backend.pupa_config import YAML_FILE  # type: ignore[import]

# ── tiny ANSI palette (matches scripts/setup.py) ────────────────────────────
_B = "\033[1m"; _D = "\033[2m"; _R = "\033[31m"; _G = "\033[32m"
_Y = "\033[33m"; _C = "\033[36m"; _X = "\033[0m"


def _ask(prompt: str, default: str = "") -> str:
    suffix = f"  {_Y}[{default}]{_X}" if default else ""
    return input(f"  {_C}{prompt}{_X}{suffix}: ").strip() or default


def _choose(prompt: str, options: list[tuple[str, str]], default: str) -> str:
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


def _ask_pairs(label: str) -> dict[str, str]:
    """Collect KEY=VALUE pairs one per line until a blank line."""
    print(f"  {_D}{label} — one KEY=VALUE per line, blank to finish. "
          f"Use ${{VAR}} to read from the process env at startup.{_X}")
    out: dict[str, str] = {}
    while True:
        raw = input(f"    {_C}KEY=VALUE{_X}: ").strip()
        if not raw:
            return out
        if "=" not in raw:
            print(f"  {_R}expected KEY=VALUE.{_X}")
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v


def _print_servers(servers: dict) -> None:
    if not servers:
        print(f"  {_D}(no MCP servers configured){_X}")
        return
    for name, entry in servers.items():
        disabled = " [disabled]" if entry.get("enabled") is False else ""
        target = entry.get("command") or entry.get("url") or "?"
        if entry.get("args"):
            target = f"{target} {' '.join(entry['args'])}"
        transport = entry.get("transport", "stdio" if "command" in entry else "streamable_http")
        print(f"  {_B}{name}{_X}{disabled}  {_D}({transport}){_X}\n      {target}")


def _restart_note() -> None:
    print(f"  {_D}Restart the backend to apply (config.yml is read at startup).{_X}")


# ── subcommands ─────────────────────────────────────────────────────────────

def cmd_list(_args: argparse.Namespace) -> int:
    _print_servers(list_servers(load_config()))
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    cfg = load_config()
    try:
        cfg = remove_server(cfg, args.name)
    except KeyError:
        print(f"  {_R}No MCP server named {args.name!r}.{_X}")
        return 1
    write_config(cfg)
    print(f"  {_G}✓ Removed {args.name!r} from {YAML_FILE}{_X}")
    _restart_note()
    return 0


def _entry_from_flags(args: argparse.Namespace) -> tuple[str, dict] | None:
    """Build (name, entry) from CLI flags, or None to fall back to interactive."""
    # Playwright preset shortcut.
    if args.playwright:
        return (args.name or "playwright"), playwright_entry()

    # Atlassian preset shortcut.
    if args.confluence_url or args.confluence_user:
        if not (args.confluence_url and args.confluence_user):
            raise SystemExit("  --confluence-url and --confluence-user must be given together.")
        name = args.name or "atlassian"
        return name, atlassian_entry(url=args.confluence_url, username=args.confluence_user)

    if not args.name:
        return None  # nothing to go on → interactive

    if not (args.command or args.url):
        raise SystemExit("  --name needs either --command (stdio) or --url (http/sse).")

    entry = build_entry(
        command=args.command,
        args=args.arg,
        url=args.url,
        transport=args.transport,
        env=_pairs_to_dict(args.env) if args.env else None,
        headers=_pairs_to_dict(args.header) if args.header else None,
        description=args.description,
    )
    return args.name, entry


def _interactive_add() -> tuple[str, dict]:
    print()
    preset = _choose(
        "What do you want to add?",
        [
            ("atlassian",  "Atlassian Confluence  (community mcp-atlassian, stdio)"),
            ("playwright", "Playwright browser     (@playwright/mcp, stdio)"),
            ("stdio",      "Custom stdio server   (command + args, e.g. npx/uvx)"),
            ("http",       "Custom remote server  (url, http/sse)"),
        ],
        default="atlassian",
    )
    print()

    if preset == "atlassian":
        name = _ask("Config name (YAML key)", default="atlassian")
        url = _ask("CONFLUENCE_URL (e.g. https://you.atlassian.net/wiki)")
        user = _ask("CONFLUENCE_USERNAME (your Atlassian email)")
        print(f"  {_D}Token is referenced as ${{CONFLUENCE_API_TOKEN}} — export it in the "
              f"backend env; it is NOT stored in config.yml.{_X}")
        return name, atlassian_entry(url=url, username=user)

    if preset == "playwright":
        name = _ask("Config name (YAML key)", default="playwright")
        print(f"  {_D}Needs Node.js + npx, plus browser binaries: "
              f"`make install-playwright` (or `npx --yes playwright install chromium`).{_X}")
        return name, playwright_entry()

    if preset == "stdio":
        name = _ask("Config name (YAML key)")
        command = _ask("Command (e.g. npx, uvx, /path/to/server)")
        args_raw = _ask("Args (space-separated, e.g. -y @scope/server /tmp)")
        env = _ask_pairs("Environment variables")
        description = _ask("Description (optional, shown in the get_tools gate)")
        return name, build_entry(
            command=command,
            args=args_raw.split() if args_raw else None,
            env=env or None,
            description=description or None,
        )

    name = _ask("Config name (YAML key)")
    url = _ask("Server URL (e.g. https://host/mcp)")
    transport = _choose(
        "Transport:",
        [("streamable_http", "Streamable HTTP"), ("sse", "Server-Sent Events")],
        default="streamable_http",
    )
    headers = _ask_pairs("Headers")
    description = _ask("Description (optional, shown in the get_tools gate)")
    return name, build_entry(
        url=url,
        transport=transport,
        headers=headers or None,
        description=description or None,
    )


def cmd_add(args: argparse.Namespace) -> int:
    chosen = _entry_from_flags(args)
    name, entry = chosen if chosen is not None else _interactive_add()

    validate_name(name)
    cfg = load_config()
    exists = name in list_servers(cfg)
    if exists and not args.force:
        print(f"  {_R}Server {name!r} already exists. Re-run with --force to overwrite.{_X}")
        return 1
    cfg = add_server(cfg, name, entry, overwrite=True)
    write_config(cfg)

    print()
    print(f"  {_G}✓ {'Updated' if exists else 'Added'} {name!r} in {YAML_FILE}{_X}")
    _print_servers({name: entry})
    if entry.get("command") == "uvx":
        print(f"  {_D}Prereqs: `uv` on PATH (MCP client deps ship with the backend).{_X}")
    _restart_note()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pupa-backend mcp",
        description="Manage MCP servers in ~/.pupa-backend/config.yml.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List configured MCP servers.").set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="Add (or update) an MCP server. No flags → interactive.")
    p_add.add_argument("--name", help="YAML key for the server (letters/digits/-/_).")
    p_add.add_argument("--command", help="stdio launch command (e.g. npx, uvx).")
    p_add.add_argument("--arg", action="append", default=[], help="One command arg (repeatable).")
    p_add.add_argument("--url", help="Remote server URL (http/sse).")
    p_add.add_argument("--transport", help="stdio | streamable_http | sse | websocket.")
    p_add.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                       help="Env var for a stdio server (repeatable). ${VAR} allowed.")
    p_add.add_argument("--header", action="append", default=[], metavar="KEY=VAL",
                       help="HTTP header for a remote server (repeatable). ${VAR} allowed.")
    p_add.add_argument("--description", help="Human one-liner shown in the get_tools gate listing.")
    p_add.add_argument("--playwright", action="store_true",
                       help="Playwright preset: @playwright/mcp stdio server (name defaults to 'playwright').")
    p_add.add_argument("--confluence-url", help="Atlassian preset: CONFLUENCE_URL.")
    p_add.add_argument("--confluence-user", help="Atlassian preset: CONFLUENCE_USERNAME.")
    p_add.add_argument("--force", action="store_true", help="Overwrite an existing server.")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="Remove an MCP server by name.")
    p_rm.add_argument("name", help="Server name to remove.")
    p_rm.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
