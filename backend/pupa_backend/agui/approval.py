"""Plain-text approval parsing shared by interactive harnesses."""

from __future__ import annotations

_APPROVE_WORDS = frozenset(
    {
        "yes", "y", "ok", "okay", "sure", "approve", "approved", "allow",
        "allowed", "go ahead", "do it", "proceed", "confirm", "confirmed",
        "yep", "yeah",
    }
)
_ALWAYS_PHRASES = (
    "always", "auto", "yes to all", "approve all", "allow all", "don't ask",
    "dont ask", "stop asking", "run freely", "yolo",
)


def interpret_always(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    return any(phrase in lowered for phrase in _ALWAYS_PHRASES)


def interpret_approval(text: str | None) -> bool:
    if not text:
        return False
    if interpret_always(text):
        return True
    lowered = text.strip().lower()
    if lowered in _APPROVE_WORDS:
        return True
    first = lowered.split(",")[0].split(".")[0].strip()
    if first in _APPROVE_WORDS:
        return True
    return any(
        phrase in lowered
        for phrase in ("go ahead", "do it", "approve", "allow it", "permission granted")
    )
