"""`pupa-backend` command-line entrypoint.

Registered as a console script (`[project.scripts]` in pyproject.toml), so a
`pip`/`pipx` install puts a `pupa-backend` executable on PATH. Replaces the old
bash wrapper that install.sh generated — this works with no cloned repo, driving
the backend and its operator scripts as in-process package imports rather than
`make` targets.

Subcommands mirror the historical CLI: run / stop / status / pair / setup / mcp /
service-* / logs / screenshare.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

PIDFILE = Path.home() / ".pupa-backend" / "pupa.pid"
LOG_DIR = Path.home() / ".pupa-backend" / "logs"
DEFAULT_BROKER = "ws://localhost:8004/screenshare/ws"
SIDECAR_TOKEN_FILE = Path("/tmp/pupa-sidecar.token")

USAGE = """Usage: pupa-backend <command> [args]

Commands:
  run, start        Start the backend (foreground)
  stop              Stop a running backend (via pidfile)
  status            Show whether the backend is running
  pair [opts]       Mint a QR pairing code for a device
  setup             Run the interactive setup wizard
  mcp <add|list|remove> [opts]   Manage MCP servers in config.yml
  service-install   Install as a launchd/systemd background service
  service-start     Start (install) the background service
  service-stop      Stop and remove the background service
  logs              Tail the service log
  screenshare       Start the macOS screen-share sidecar (source build)
"""


def _load_config() -> None:
    """Apply ~/.pupa-backend/config.yml into os.environ. Shell-exported
    values already present win (load_pupa_config does not overwrite them)."""
    from pupa_backend.pupa_config import load_pupa_config

    load_pupa_config()


# ── run / stop / status ──────────────────────────────────────────────────────

def _run() -> int:
    _load_config()
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    try:
        from pupa_backend.app import main as app_main

        app_main()
    finally:
        try:
            if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except OSError:
            pass
    return 0


def _read_pid() -> int | None:
    try:
        pid = int(PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # liveness probe; does not actually signal
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid  # alive but not ours to signal
    return pid


def _stop() -> int:
    pid = _read_pid()
    if pid is None:
        print("Backend not running.")
        PIDFILE.unlink(missing_ok=True)
        return 0
    os.kill(pid, signal.SIGTERM)
    PIDFILE.unlink(missing_ok=True)
    print(f"Backend stopped (pid {pid}).")
    return 0


def _status() -> int:
    print("Running" if _read_pid() is not None else "Stopped")
    return 0


# ── operator scripts (in-process) ────────────────────────────────────────────

def _forward(label: str, rest: list[str], target) -> int:
    """Call a script's argparse `main()`, which reads sys.argv. Splice `rest`
    in as its argv so the script's own flags forward unchanged."""
    saved = sys.argv
    sys.argv = [f"pupa-backend {label}", *rest]
    try:
        target()
        return 0
    except SystemExit as exc:  # argparse / sys.exit inside the script
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = saved


def _pair(rest: list[str]) -> int:
    from pupa_backend.scripts.pair import main as pair_main

    if "BACKEND_URL" not in os.environ:
        _load_config()
        scheme = "https" if os.environ.get("PUPA_TLS_CERT") else "http"
        os.environ.setdefault("BACKEND_URL", f"{scheme}://localhost:8004")
    return _forward("pair", rest, pair_main)


def _setup(rest: list[str]) -> int:
    from pupa_backend.scripts.setup import main as setup_main

    return _forward("setup", rest, setup_main)


def _mcp(rest: list[str]) -> int:
    from pupa_backend.scripts.mcp import main as mcp_main

    return _forward("mcp", rest, mcp_main)


def _service(action: str) -> int:
    from pupa_backend.scripts import service

    {"install": service.install, "start": service.install,
     "stop": service.uninstall}[action]()
    return 0


# ── logs / screenshare / install-mcp ─────────────────────────────────────────

def _logs() -> int:
    logs = sorted(LOG_DIR.glob("*.log"))
    if not logs:
        print(f"No service logs at {LOG_DIR}")
        print("Use 'pupa-backend run' to start in the foreground instead.")
        return 1
    try:
        subprocess.run(["tail", "-f", *map(str, logs)])
    except KeyboardInterrupt:
        pass
    return 0


def _screenshare() -> int:
    if sys.platform != "darwin":
        print("screenshare is macOS-only — not available on this platform.")
        return 1
    sidecar = os.environ.get("PUPA_SIDECAR_DIR")
    sidecar_dir = Path(sidecar) if sidecar else Path.cwd() / "screenshare-sidecar"
    if not sidecar_dir.is_dir():
        print(
            "Screen-share sidecar is a separate Swift package, built from source.\n"
            "It is not shipped in the pip package. To use it:\n\n"
            "  git clone https://github.com/pupa-app/pupa-backend.git\n"
            "  swift build --package-path pupa-backend/screenshare-sidecar\n\n"
            "Then point the CLI at it:\n"
            "  PUPA_SIDECAR_DIR=/path/to/pupa-backend/screenshare-sidecar pupa-backend screenshare"
        )
        return 1
    from shutil import which

    if not which("swift"):
        print("Swift not found — install Xcode or the Swift toolchain first.")
        return 1
    if not SIDECAR_TOKEN_FILE.exists():
        print(f"Sidecar token not found at {SIDECAR_TOKEN_FILE} — start the backend first.")
        return 1
    token = SIDECAR_TOKEN_FILE.read_text().strip()
    broker = os.environ.get("SCREENSHARE_BROKER", DEFAULT_BROKER)
    share_id = os.environ.get("SHARE_ID") or str(uuid.uuid4())
    print(f"  share id: {share_id}")
    return subprocess.run([
        "swift", "run", "--package-path", str(sidecar_dir), "pupa-screenshare",
        "--broker", broker, "--share-id", share_id, "--api-key", token,
    ]).returncode


def _install_mcp() -> int:
    print("MCP client deps ship in the core package now — nothing to install.")
    return 0


# ── dispatch ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]

    if cmd in ("run", "start"):
        return _run()
    if cmd == "stop":
        return _stop()
    if cmd == "status":
        return _status()
    if cmd == "pair":
        return _pair(rest)
    if cmd == "setup":
        return _setup(rest)
    if cmd == "mcp":
        return _mcp(rest)
    if cmd == "service-install":
        return _service("install")
    if cmd == "service-start":
        return _service("start")
    if cmd == "service-stop":
        return _service("stop")
    if cmd == "logs":
        return _logs()
    if cmd == "screenshare":
        return _screenshare()
    if cmd == "install-mcp":
        return _install_mcp()

    print(f"Unknown command: {cmd}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
