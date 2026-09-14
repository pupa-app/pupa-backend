"""Normalise AG-UI messages and ambient context for every harness."""

from __future__ import annotations

import json
from typing import Any


def message_content(message: Any) -> Any:
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content


def message_role(message: Any) -> str | None:
    return getattr(message, "role", None) or (
        message.get("role") if isinstance(message, dict) else None
    )


def coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif getattr(part, "text", None):
                parts.append(part.text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def latest_user_message(messages: list[Any]) -> Any:
    for message in reversed(messages or []):
        if message_role(message) == "user":
            return message
    return None


def latest_user_text(messages: list[Any]) -> str:
    return coerce_content(message_content(latest_user_message(messages)))


def render_transcript(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages or []:
        text = coerce_content(message_content(message))
        if text:
            lines.append(f"{message_role(message) or 'unknown'}: {text}")
    return "\n\n".join(lines)


def canonical_json(text: str) -> str:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(parsed, (dict, list)):
        return text
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def context_pairs(context: list[Any] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for entry in context or []:
        description = getattr(entry, "description", None)
        value = getattr(entry, "value", None)
        if isinstance(entry, dict):
            description = description if description is not None else entry.get("description")
            value = value if value is not None else entry.get("value")
        pairs.append(((description or "").strip(), canonical_json((value or "").strip())))
    return pairs


def render_context(context: list[Any] | None) -> str:
    blocks: list[str] = []
    for description, value in context_pairs(context):
        block = f"{description}\n{value}".strip()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def image_inputs(content: Any) -> list[dict[str, Any]]:
    """Convert AG-UI image parts to Codex App Server user-input objects."""
    if not isinstance(content, list):
        return []
    images: list[dict[str, Any]] = []
    for part in content:
        part_type = getattr(part, "type", None) or (
            part.get("type") if isinstance(part, dict) else None
        )
        if part_type != "image":
            continue
        source = getattr(part, "source", None)
        if source is None and isinstance(part, dict):
            source = part.get("source")
        if source is None:
            continue
        source_type = getattr(source, "type", None)
        value = getattr(source, "value", None)
        mime_type = getattr(source, "mime_type", None)
        if isinstance(source, dict):
            source_type = source_type or source.get("type")
            value = value or source.get("value")
            mime_type = mime_type or source.get("mime_type") or source.get("mimeType")
        if not value:
            continue
        url = str(value)
        if source_type != "url":
            url = f"data:{mime_type or 'image/jpeg'};base64,{value}"
        images.append({"type": "image", "url": url})
    return images
