"""Regression suite for the backend skill lifecycle (foundation).

Pins the foundation contracts of `skills.py` / `PupaSkillsMiddleware`:

1. **Discovery, not disclosure.** At agent start a skill's *name and
   description* are injected into the system prompt, but its full SKILL.md
   *body* is NOT — that's the whole point of progressive disclosure.

2. **`skill_view` is the read stage.** The `skill_view` tool is bound to the
   agent and returns the full SKILL.md body for a known skill; unknown names and
   path traversal attempts are rejected without touching the filesystem.

3. **On by default, opt-out.** The `skills` spec ships on; `enabled_specs()`
   includes it unless `PUPA_SKILLS_DISABLED` is set.

Tests point the skills root at a `tmp_path` via `PUPA_SKILLS_DIR`, so they run
without a shipped skill and without writing to the real `~/.pupa-backend`.
`MockChatModel` from conftest keeps them credential-free.
"""



from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from pupa_backend.harnesses.langgraph.skills import _build_skill_view_tool, build_skills_middlewares

from .conftest import MockChatModel


_SKILL_BODY = """---
name: web-research
description: Decompose into sub-questions, search, cross-check sources, synthesize.
---

# Web research

## Workflow
1. Cross-check every load-bearing fact against a second source.
"""


def _write_skill(root: Path, name: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(root, "web-research", _SKILL_BODY)
    monkeypatch.setenv("PUPA_SKILLS_DIR", str(root))
    return root


class _CollectingMiddleware(AgentMiddleware):
    """Captures the system message of each model call it wraps."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def awrap_model_call(self, request, handler):
        self._sink.append(str(request.system_message))
        return await handler(request)


async def test_discovery_injects_name_and_description_not_body(skills_root):
    """The skill name + description reach the system prompt; the body does not."""
    captured: list[str] = []
    skills_mw = build_skills_middlewares()
    assert skills_mw, "expected the skills middleware to be built"

    model = MockChatModel(responses=[AIMessage(content="ok", id="m1")])
    agent = create_agent(
        model=model,
        tools=[],
        # skills middleware first (outer) so the collector (inner) sees the
        # system prompt it injected.
        middleware=[*skills_mw, _CollectingMiddleware(captured)],
        checkpointer=MemorySaver(),
        name="skills_discovery_test",
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi", id="h1")]},
        config={"configurable": {"thread_id": "skills-discovery"}},
    )

    assert len(captured) == 1
    system = captured[0]
    # Discovery: name + a distinctive phrase from the description.
    assert "web-research" in system
    assert "Decompose into sub-questions" in system
    # Disclosure withheld: a distinctive phrase from the SKILL.md body must NOT
    # be present until the agent reads the skill.
    assert "# Web research" not in system
    assert "load-bearing fact" not in system


async def test_skill_view_tool_is_bound(skills_root):
    """`skill_view` is carried on the middleware and bound to the agent."""
    collected: list[list[str]] = []
    skills_mw = build_skills_middlewares()

    class _ToolCollector(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            collected.append([
                (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
                for t in request.tools
            ])
            return await handler(request)

    model = MockChatModel(responses=[AIMessage(content="ok", id="m1")])
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[*skills_mw, _ToolCollector()],
        checkpointer=MemorySaver(),
        name="skills_tool_test",
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi", id="h1")]},
        config={"configurable": {"thread_id": "skills-tool"}},
    )

    assert len(collected) == 1
    assert "skill_view" in collected[0]


def test_skill_view_returns_body_for_known_skill(skills_root):
    skill_view = _build_skill_view_tool(skills_root)
    body = skill_view.invoke({"name": "web-research"})
    assert "# Web research" in body
    assert "Cross-check" in body


def test_skill_view_rejects_unknown_and_traversal(skills_root, tmp_path):
    skill_view = _build_skill_view_tool(skills_root)
    assert skill_view.invoke({"name": "does-not-exist"}).startswith("Unknown skill")
    # Path traversal / separators must be rejected before any filesystem read.
    assert skill_view.invoke({"name": "../skills"}).startswith("Unknown skill")
    assert skill_view.invoke({"name": "../../etc/passwd"}).startswith("Unknown skill")
    assert skill_view.invoke({"name": ""}).startswith("Unknown skill")

    # A real file outside the skills root must not be reachable even if a
    # sibling skill dir of that name does not exist.
    secret = tmp_path / "secret.md"
    secret.write_text("top secret", encoding="utf-8")
    assert skill_view.invoke({"name": "../secret.md"}).startswith("Unknown skill")


def test_empty_skills_root_is_valid(tmp_path, monkeypatch):
    """A fresh install (empty root) builds the middleware without error and
    lists no skills."""
    monkeypatch.setenv("PUPA_SKILLS_DIR", str(tmp_path / "skills"))
    skills_mw = build_skills_middlewares()
    assert skills_mw, "middleware should build even with no skills present"


def test_skills_spec_on_by_default(monkeypatch):
    """The `skills` spec is enabled unless PUPA_SKILLS_DISABLED is set."""
    import pupa_backend.harnesses.langgraph.backend_tools as backend_tools

    monkeypatch.delenv("PUPA_SKILLS_DISABLED", raising=False)
    names = {spec.name for spec in backend_tools.enabled_specs()}
    assert "skills" in names

    monkeypatch.setenv("PUPA_SKILLS_DISABLED", "1")
    names = {spec.name for spec in backend_tools.enabled_specs()}
    assert "skills" not in names
