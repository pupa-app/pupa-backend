"""Backend agent skill lifecycle — progressive disclosure over user skills.

Wraps deepagents' `SkillsMiddleware` (the Agent Skills spec loader/parser) to
give the Pupa agent a first-class skill unit, following the standard Agent
Skills progressive-disclosure shape:

  - Skills live under a single user-writable root, `~/.pupa-backend/skills/`
    (next to `config.yml`, `checkpoints.db`, `store.db`). The package ships NO
    built-in skills — the directory starts empty and is populated by the user
    or, later, a marketplace install path.
  - Each immediate subdirectory is one skill: a `SKILL.md` with YAML
    frontmatter (`name`, `description`, optional `metadata` / `allowed-tools`).

Progressive disclosure — two stages in this foundation:

  1. DISCOVERY — `SkillsMiddleware` injects every skill's *name + description*
     into the system prompt at agent start. Full bodies stay out of the prompt.
     (A compact skill index is injected rather than making the model blind-call
     a list tool, because models don't reliably reach for such a tool.)
  2. READ — the agent calls the read-only `skill_view` tool with a skill name
     to pull that skill's full `SKILL.md` body on demand.

We deliberately do NOT mount deepagents' `FilesystemMiddleware` for the read
stage: it registers seven read/write/execute tools on every turn, which both
bloats the per-turn token payload (the opposite of what progressive disclosure
is for) and hands the model a general-purpose filesystem surface. A single
scoped, read-only `skill_view` tool covers the read stage and cannot escape the
skills directory.

Default ON; opt out with `PUPA_SKILLS_DISABLED=1`. Wired through
`backend_tools.py` like every other optional backend capability.

Skills are loaded once per session (`SkillsMiddleware` caches them in agent
state), so a new or edited skill only appears in a fresh session — the iOS
client mints one `thread_id` per New Session.

Future work: per-skill tool gating, a LangGraph store backend for
per-device installed skills, frontend MyApp parity (provenance
labels + no-orphan session gating), and a marketplace `install_skill` path.
"""

import logging
import os
from pathlib import Path
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillMetadata, SkillsMiddleware
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger("uvicorn.error")

# Override env var for the skills root — handy for tests and non-default homes.
SKILLS_DIR_ENV = "PUPA_SKILLS_DIR"
# Default user-writable skills root, alongside the rest of the backend's state.
DEFAULT_SKILLS_DIR = Path.home() / ".pupa-backend" / "skills"


def skills_dir() -> Path:
    """Resolve the skills root: `PUPA_SKILLS_DIR` if set, else the default."""
    raw = os.getenv(SKILLS_DIR_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_SKILLS_DIR


# System-prompt fragment. Must keep the three `{...}` slots `SkillsMiddleware`
# substitutes at request time. Phrased around `skill_view` (not deepagents'
# default `read_file`) because that's the only read mechanism we expose.
_SKILLS_SYSTEM_PROMPT = """## Skills

You have a library of skills — reusable, specialized workflows. You see each
skill's name and description below, but NOT its full instructions; that keeps
your context lean (progressive disclosure).
{skills_locations}{skills_load_warnings}

**Available skills:**

{skills_list}

**How to use a skill:**

1. Match the user's task against a skill's description.
2. Call `skill_view` with the skill's `name` to load its full instructions.
3. Follow them.

Read a skill only when its description matches the task at hand — don't read
skills speculatively."""


class PupaSkillsMiddleware(SkillsMiddleware):
    """`SkillsMiddleware` variant that drives the read stage through the scoped
    `skill_view` tool instead of deepagents' filesystem `read_file`.

    Overrides the per-skill list formatting so the prompt never leaks backend
    filesystem paths or points the model at a `read_file` tool it doesn't have.
    The `skill_view` tool is carried on `self.tools` so `create_agent` binds it.
    """

    def __init__(self, *, skills_dir: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tools: list[BaseTool] = [_build_skill_view_tool(skills_dir)]

    def _format_skills_locations(self) -> str:
        # Suppress deepagents' "**Skills**: `/`" line — the virtual backend path
        # is meaningless to the model, and the read path is the skill name via
        # `skill_view`, not a filesystem location.
        return ""

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        if not skills:
            return "(No skills available.)"
        lines: list[str] = []
        for skill in skills:
            line = f"- **{skill['name']}**: {skill['description']}"
            if skill.get("allowed_tools"):
                line += f" (uses: {', '.join(skill['allowed_tools'])})"
            lines.append(line)
        return "\n".join(lines)


def _build_skill_view_tool(root_dir: Path) -> BaseTool:
    """Build the read-only `skill_view` tool bound to `root_dir`.

    The tool reads `<root_dir>/<name>/SKILL.md`. `name` is validated to be a
    direct child directory (no separators, no leading dot) and the resolved path
    is confirmed to stay inside `root_dir`, so the tool cannot read arbitrary
    files via traversal or absolute paths.
    """
    root = root_dir.resolve()

    @tool
    def skill_view(name: str) -> str:
        """Load the full instructions for a skill by name.

        Call this only after a skill's description (listed in your system
        prompt) matches the user's task. Returns the skill's SKILL.md body.
        """
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return f"Unknown skill {name!r}."
        skill_md = root / name / "SKILL.md"
        try:
            resolved = skill_md.resolve()
            resolved.relative_to(root)
        except (ValueError, OSError):
            return f"Unknown skill {name!r}."
        if not resolved.is_file():
            return f"Unknown skill {name!r}."
        return resolved.read_text(encoding="utf-8")

    return skill_view


def build_skills_middlewares() -> list[Any]:
    """Factory consumed by `backend_tools.py` (default on; off when
    `PUPA_SKILLS_DISABLED=1`).

    Returns `[PupaSkillsMiddleware]`. The middleware loads skills from the user
    skills root for discovery and carries the `skill_view` tool for the read
    stage. The root is created if absent, so a fresh install starts with an
    empty (but valid) skills library rather than erroring.
    """
    root = skills_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("[pupa] cannot create skills dir %s: %s; skills disabled.", root, exc)
        return []

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    middleware = PupaSkillsMiddleware(
        skills_dir=root,
        backend=backend,
        sources=["/"],
        system_prompt=_SKILLS_SYSTEM_PROMPT,
    )
    return [middleware]
