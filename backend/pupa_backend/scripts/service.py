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
from pathlib import Path

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


# ---------------------------------------------------------------------------
# macOS — launchd LaunchAgent
# ---------------------------------------------------------------------------

def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _launchd_plist(backend_dir: Path, python: Path) -> str:
    log_dir = _log_dir()

    # Load config from config.yml (or .env legacy) without applying to os.environ.
    from pupa_backend.pupa_config import load_pupa_config  # type: ignore[import]
    env_dict = load_pupa_config(apply=False)
    env_dict["PATH"] = _service_path(env_dict)

    env_vars = ""
    for key, val in env_dict.items():
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
    from pupa_backend.pupa_config import load_pupa_config  # type: ignore[import]
    env_dict = load_pupa_config(apply=False)
    env_dict["PATH"] = _service_path(env_dict)
    env_lines = "".join(_systemd_env_line(k, v) for k, v in env_dict.items())

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
