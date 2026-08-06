"""`python -m pupa_backend` → the CLI dispatcher.

Note the server itself is launched with `python -m pupa_backend.app` (see the
Makefile / Dockerfile / systemd unit), which runs `app.py`'s `main()` directly.
"""

from pupa_backend.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
