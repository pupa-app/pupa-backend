"""Env-var gating for the screen-share feature."""

import os


def is_enabled() -> bool:
    return os.getenv("PUPA_SCREENSHARE", "").strip() not in ("", "0", "false", "False")
