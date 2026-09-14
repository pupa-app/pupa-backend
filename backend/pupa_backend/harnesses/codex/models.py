"""Runtime-discovered Codex model and reasoning menus."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("uvicorn.error")

_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max", "ultra")


@dataclass
class ModelCatalog:
    data: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> "ModelCatalog":
        rows = response.get("data") or []
        return cls([row for row in rows if isinstance(row, dict) and not row.get("hidden")])

    def menu(self) -> list[dict[str, str]]:
        return [
            {
                "provider": "codex",
                "modelId": str(row.get("model") or row.get("id")),
                "label": str(row.get("displayName") or row.get("model") or row.get("id")),
            }
            for row in self.data
            if row.get("model") or row.get("id")
        ]

    def thinking_menu(self) -> list[dict[str, str]]:
        available = {
            str(option.get("reasoningEffort"))
            for row in self.data
            for option in row.get("supportedReasoningEfforts") or []
            if isinstance(option, dict) and option.get("reasoningEffort")
        }
        ordered = [value for value in _EFFORT_ORDER if value in available]
        ordered.extend(sorted(available - set(ordered)))
        return [{"level": value, "label": value.replace("xhigh", "Extra high").title()} for value in ordered]

    def default_model(self) -> str | None:
        configured = (os.getenv("PUPA_CODEX_MODEL") or "").strip()
        if configured and self.has(configured):
            return configured
        for row in self.data:
            if row.get("isDefault"):
                return str(row.get("model") or row.get("id"))
        return str(self.data[0].get("model") or self.data[0].get("id")) if self.data else None

    def has(self, model: str) -> bool:
        return any(model in {str(row.get("model")), str(row.get("id"))} for row in self.data)

    def row(self, model: str) -> dict[str, Any] | None:
        return next(
            (row for row in self.data if model in {str(row.get("model")), str(row.get("id"))}),
            None,
        )

    def resolve_model(self, run_input: Any) -> str | None:
        forwarded = getattr(run_input, "forwarded_props", None) or {}
        llm = forwarded.get("llm") if isinstance(forwarded, dict) else None
        requested = llm.get("model") if isinstance(llm, dict) else None
        if requested and self.has(str(requested)):
            return str(requested)
        if requested:
            logger.warning("codex harness: unsupported model %r; using account default", requested)
        return self.default_model()

    def resolve_effort(self, run_input: Any, model: str | None) -> str | None:
        forwarded = getattr(run_input, "forwarded_props", None) or {}
        llm = forwarded.get("llm") if isinstance(forwarded, dict) else None
        requested = llm.get("thinking") if isinstance(llm, dict) else None
        row = self.row(model) if model else None
        supported = {
            str(option.get("reasoningEffort"))
            for option in (row or {}).get("supportedReasoningEfforts") or []
            if isinstance(option, dict) and option.get("reasoningEffort")
        }
        if requested in supported:
            return str(requested)
        if requested:
            logger.warning(
                "codex harness: reasoning effort %r is unsupported by %r; using model default",
                requested,
                model,
            )
        default = (row or {}).get("defaultReasoningEffort")
        return str(default) if default else None
