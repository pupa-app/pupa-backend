"""Shared AG-UI and LangChain tool normalisation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


def descriptor_fields(descriptor: Any) -> tuple[str | None, str, dict[str, Any]]:
    get = (
        (lambda key: getattr(descriptor, key, None))
        if not isinstance(descriptor, dict)
        else descriptor.get
    )
    name = get("name")
    description = get("description") or ""
    schema = get("parameters") or get("input_schema") or get("inputSchema")
    if name is None and isinstance(descriptor, dict) and isinstance(descriptor.get("function"), dict):
        function = descriptor["function"]
        name = function.get("name")
        description = function.get("description") or description
        schema = function.get("parameters") or schema
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    return name, description, schema


def descriptor_specs(descriptors: list[Any]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for descriptor in descriptors or []:
        name, description, schema = descriptor_fields(descriptor)
        if name:
            specs.append(ToolSpec(str(name), description, schema))
    return specs


def langchain_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            value = model_json_schema()
            if isinstance(value, dict):
                return value
        except Exception:  # noqa: BLE001 - permissive schema is a safe fallback
            pass
    return {"type": "object", "properties": {}}


def langchain_tool_specs(mcp: Any) -> list[tuple[Any, ToolSpec]]:
    result: list[tuple[Any, ToolSpec]] = []
    for tool in list(getattr(mcp, "tools", None) or []) if mcp is not None else []:
        name = getattr(tool, "name", None)
        if name:
            result.append(
                (
                    tool,
                    ToolSpec(
                        str(name),
                        getattr(tool, "description", "") or "",
                        langchain_input_schema(tool),
                    ),
                )
            )
    return result


async def invoke_langchain_tool(tool: Any, arguments: dict[str, Any]) -> tuple[bool, str]:
    try:
        result = await tool.ainvoke(arguments or {})
    except Exception as exc:  # noqa: BLE001 - tool errors belong in model context
        return False, f"Error calling tool: {exc}"
    if isinstance(result, str):
        return True, result
    try:
        return True, json.dumps(result, default=str)
    except (TypeError, ValueError):
        return True, str(result)
