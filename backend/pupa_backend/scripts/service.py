"""Install/uninstall the Pupa backend as an OS service.

macOS : launchd LaunchAgent in ~/Library/LaunchAgents/
Linux : systemd user service in ~/.config/systemd/user/

The service auto-starts on login and restarts on failure. Logs go to
~/.pupa-backend/logs/.

Run via:
  make service-install    — install and start
  make service-uninstall  — stop and remove
  make service-status     — show status
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from pupa_backend.pupa_config import known_env_vars

LABEL = "com.pupa.backend"
SERVICE_NAME = "pupa-backend"


def _backend_dir() -> Path:
    return Path(__file__).parent.parent.resolve()


def _service_python() -> Path:
    """Interpreter the service should exec. `sys.executable` is the Python
    running this install command — the pipx (or source venv) interpreter that
    has `pupa_backend` importable — so `-m pupa_backend.app` resolves without
    depending on a cloned repo or a `.venv/` next to the package."""
    return Path(sys.executable)


def _log_dir() -> Path:
    d = Path.home() / ".pupa-backend" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _service_path(env_dict: dict[str, str]) -> str:
    """PATH for the service process.

    systemd/launchd start a service with a minimal PATH that excludes
    ~/.local/bin, Homebrew, nvm, etc. The claude_code agent loop shells
    out to `claude`, and MCP servers to `uvx`/`node`, so carry the
    installing user's PATH (plus the dir the `claude` binary resolves to
    on PATH) into the unit — otherwise startup aborts with
    `claude not found on PATH`.
    """
    path = env_dict.get("PATH") or os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    claude = shutil.which("claude")
    if claude:
        cdir = str(Path(claude).parent)  # dir on PATH, not the symlink target
        if cdir not in parts:
            parts.insert(0, cdir)
    return os.pathsep.join(parts)


def _unit_env() -> dict[str, str]:
    """The env the generated unit carries. PATH, and nothing else.

    `app.py` calls `load_pupa_config()` at import, so the service process reads
    config.yml itself — snapshotting those values into the unit was redundant and
    actively harmful. The plist is written 0644 while config.yml is 0600, so the
    snapshot copied every secret into a world-readable file; and it froze at
    install time, so editing config.yml did nothing until a reinstall.

    PATH is the exception: launchd/systemd start with a minimal PATH and the
    backend cannot reconstruct the operator's, so it has to be passed in.
    """
    from pupa_backend.pupa_config import load_pupa_config  # type: ignore[import]

    # A PATH set under `env:` in config.yml wins; otherwise the installing
    # process's own PATH is carried over.
    return {"PATH": _service_path(load_pupa_config(apply=False))}


# ---------------------------------------------------------------------------
# Shell-only credential guard
# ---------------------------------------------------------------------------

_BYPASS_VAR = "PUPA_SERVICE_ALLOW_SHELL_ONLY"


def _check_env_names(data: dict) -> list[str]:
    """Operator-supplied extra var names from `service.check_env` in config.yml.

    `known_env_vars()` covers everything the config *schema* can name. Vars that
    only exist under the `env:` passthrough (raw AWS keys, an MCP server's token,
    a third-party SDK's var) have no schema entry, so an operator lists them here
    to have the install guard watch them too.
    """
    block = data.get("service")
    if not isinstance(block, dict):
        return []
    names = block.get("check_env")
    if not isinstance(names, list):
        return []
    return [str(n) for n in names if str(n).strip()]


def _shell_only_secrets(
    env_dict: dict[str, str],
    extra: Sequence[str] = (),
) -> list[str]:
    """Vars set in the installing shell but absent from config.yml.

    The candidate set is derived from `pupa_config.known_env_vars()` — the same
    maps the loader writes through — plus `extra`, so nothing is hand-maintained
    here. A var outside both is not reported: config.yml could not have set it
    anyway, so there would be no fix to offer.
    """
    candidates = set(known_env_vars()) | set(extra)
    return sorted(
        name
        for name in candidates
        if os.environ.get(name) and not (env_dict.get(name) or "").strip()
    )


def _assert_no_shell_only_secrets(
    env_dict: dict[str, str],
    extra: Sequence[str] = (),
) -> None:
    """Abort the install rather than write a unit that will crash-loop.

    `pupa-backend run` inherits the shell; a launchd agent / systemd unit does
    not. Failing here — with the terminal still in front of the operator — beats
    failing at startup inside a log file.
    """
    missing = _shell_only_secrets(env_dict, extra)
    if not missing or os.environ.get(_BYPASS_VAR):
        return

    from pupa_backend.pupa_config import YAML_FILE  # type: ignore[import]

    schema = known_env_vars()
    width = max(len(n) for n in missing)
    lines = [
        "Refusing to install: these vars are set in your shell but missing from",
        f"{YAML_FILE}:",
        "",
    ]
    for name in missing:
        # Vars outside the schema have no typed home — they go under `env:`.
        hint = schema.get(name) or f"env.{name}"
        lines.append(f"  {name:<{width}}  ->  config.yml: {hint}")
    lines += [
        "",
        "A background service does not inherit your shell environment. `pupa-backend",
        "run` works because it is a child of your terminal; the service is not, so",
        "anything exported in ~/.zshrc or ~/.bashrc is invisible to it. Put these",
        "values in config.yml (mode 0600), then re-run service-install.",
        "",
        f"To install anyway (the service starts without them): {_BYPASS_VAR}=1",
    ]
    sys.exit("\n".join(lines))


def _guard_install_env() -> None:
    """Run the shell-only guard against the operator's real config.yml."""
    from pupa_backend.pupa_config import load_pupa_config, load_raw_config  # type: ignore[import]

    _assert_no_shell_only_secrets(
        load_pupa_config(apply=False),
        extra=_check_env_names(load_raw_config()),
    )


# ---------------------------------------------------------------------------
# macOS — launchd LaunchAgent
# ---------------------------------------------------------------------------

def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _launchd_plist(backend_dir: Path, python: Path) -> str:
    log_dir = _log_dir()

    # PATH only — see `_unit_env`. Everything else is read from config.yml by the
    # service process itself, at startup.
    env_vars = ""
    for key, val in _unit_env().items():
        val = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        env_vars += f"        <key>{key}</key>\n        <string>{val}</string>\n"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>pupa_backend.app</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{backend_dir}</string>

    <key>EnvironmentVariables</key>
    <dict>
{env_vars}    </dict>

    <key>StandardOutPath</key>
    <string>{log_dir / "backend.log"}</string>

    <key>StandardErrorPath</key>
    <string>{log_dir / "backend.log"}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>Crashed</key>
        <true/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>5</integer>
</dict>
</plist>
"""


def _install_launchd(backend_dir: Path) -> None:
    _guard_install_env()

    python = _service_python()

    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(_launchd_plist(backend_dir, python))
    plist_path.chmod(0o644)

    # Unload first in case a stale entry exists.
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"launchctl load failed:\n{result.stderr}")

    print(f"  Service installed and started: {plist_path}")
    print(f"  Logs: {_log_dir() / 'backend.log'}")
    print(f"  Manage: launchctl [start|stop|unload] {LABEL}")


def _uninstall_launchd() -> None:
    plist_path = _launchd_plist_path()
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    if plist_path.exists():
        plist_path.unlink()
        print(f"  Service removed: {plist_path}")
    else:
        print("  Service not installed (plist not found).")


def _status_launchd() -> None:
    result = subprocess.run(
        ["launchctl", "list", LABEL],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"  Service '{LABEL}' is not loaded.")


# ---------------------------------------------------------------------------
# Linux — systemd user service
# ---------------------------------------------------------------------------

def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"


def _systemd_env_line(key: str, value: str) -> str:
    """Render one `Environment=` line, quoted so it survives systemd parsing.

    systemd splits an unquoted value on whitespace, so any value with
    spaces (e.g. the PUPA_MCP_SERVERS JSON blob) must be quoted. Wrap the
    whole assignment in double quotes and escape backslash, double-quote,
    and percent (a specifier char) so JSON round-trips intact.
    """
    esc = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{key}={esc}"\n'


def _systemd_unit(backend_dir: Path, python: Path) -> str:
    env_lines = "".join(_systemd_env_line(k, v) for k, v in _unit_env().items())

    # StartLimit* belong in [Unit], not [Service] — systemd ignores them in
    # [Service], so the restart limiter would never engage on a crash loop.
    return f"""[Unit]
Description=Pupa backend
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
ExecStart={python} -m pupa_backend.app
WorkingDirectory={backend_dir}
Restart=always
RestartSec=5
StandardOutput=append:{_log_dir() / "backend.log"}
StandardError=append:{_log_dir() / "backend.log"}
{env_lines}
[Install]
WantedBy=default.target
"""


def _install_systemd(backend_dir: Path) -> None:
    if not shutil.which("systemctl"):
        sys.exit("systemctl not found — is systemd running?")

    _guard_install_env()

    python = _service_python()

    unit_path = _systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(_systemd_unit(backend_dir, python))

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", SERVICE_NAME], check=True)

    print(f"  Service installed and started: {unit_path}")
    print(f"  Logs: journalctl --user -u {SERVICE_NAME} -f")
    print(f"  Manage: systemctl --user [start|stop|restart|status] {SERVICE_NAME}")


def _uninstall_systemd() -> None:
    subprocess.run(["systemctl", "--user", "disable", "--now", SERVICE_NAME], capture_output=True)
    unit_path = _systemd_unit_path()
    if unit_path.exists():
        unit_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    print(f"  Service removed: {unit_path}")


def _status_systemd() -> None:
    subprocess.run(["systemctl", "--user", "status", SERVICE_NAME])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _system() -> str:
    return platform.system()


def install(backend_dir: Path | None = None) -> None:
    # Neutral working dir: config is read from ~/.pupa-backend, and the app
    # runs via `-m pupa_backend.app` from site-packages, so we must not cwd
    # into the package dir (that would put bare module names back on sys.path).
    bd = backend_dir or Path.home()
    if _system() == "Darwin":
        _install_launchd(bd)
    elif _system() == "Linux":
        _install_systemd(bd)
    else:
        sys.exit(f"Service installation not supported on {_system()}.")


def uninstall() -> None:
    if _system() == "Darwin":
        _uninstall_launchd()
    elif _system() == "Linux":
        _uninstall_systemd()
    else:
        sys.exit(f"Service uninstall not supported on {_system()}.")


def status() -> None:
    if _system() == "Darwin":
        _status_launchd()
    elif _system() == "Linux":
        _status_systemd()
    else:
        sys.exit(f"Service status not supported on {_system()}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage the Pupa backend service.")
    parser.add_argument("action", choices=["install", "uninstall", "status"])
    args = parser.parse_args()

    if args.action == "install":
        install()
    elif args.action == "uninstall":
        uninstall()
    else:
        status()
