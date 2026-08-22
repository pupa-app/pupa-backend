"""Central registry of backend tools (and tool-providing middlewares).

Each `BackendToolSpec` declares a tool's wire name, a one-line description for
the iOS Settings sheet, the env var that gates whether the backend can attach
it at all (e.g. an API key, a boolean opt-in), and a factory that materialises
either a LangChain tool (`factory`) or a LangChain agent middleware that
registers its own tool (`middleware_factory`). A spec uses exactly one of the
two.

The split between "enabled by env" (registry-side, server-side concern) and
"enabled by the user" (client-side via Settings, communicated per-turn via
`RunAgentInput.state["disabled_tools"]`) is intentional — without the gate
the tool can never run, and the discovery endpoint reflects that so the iOS
Settings sheet can grey it out. The user toggle then lives on top of that.
"""



import os
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, List

from deepagents.backends.state import StateBackend
from deepagents.middleware import SubAgentMiddleware
from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain_tavily import TavilySearch


@dataclass(frozen=True)
class BackendToolSpec:
    name: str
    description: str
    env_var: str | None
    factory: Callable[[], Any] | None = None
    middleware_factory: Callable[[], Any] | None = None
    disable_env_var: str | None = None
    # Real runtime tool name(s) this spec gates, when they differ from `name`.
    # The Settings sheet shows `name` as the disable id, but a middleware spec
    # registers tools under their own names (e.g. `subagents` → `task`). The
    # client mutes by `name`; `ToolGatingMiddleware` maps it to these names so
    # the right tools actually drop. Empty means `name` is itself the tool name.
    gated_tool_names: tuple[str, ...] = ()

    @property
    def enabled_by_env(self) -> bool:
        # An opt-out gate wins over everything: if `disable_env_var` is set in
        # the env, the spec is disabled regardless of `env_var`. This lets a
        # spec default ON (`env_var=None`) while still being switchable off.
        if self.disable_env_var is not None and os.getenv(self.disable_env_var):
            return False
        if self.env_var is None:
            return True
        return bool(os.getenv(self.env_var))


_SHELL_STARTUP_DEFAULT = os.path.join(os.path.dirname(__file__), "shell_startup.local.sh")


def _shell_env_exclude() -> frozenset[str]:
    """Build the set of env var names to strip from the shell subprocess.

    Two sources, merged:
    - ``SHELL_ENV_EXCLUDE``: comma-separated list of names.
    - ``SHELL_ENV_EXCLUDE_FROM``: path to a shell file (e.g. ``~/.zshrc``);
      every ``export VAR=`` line contributes its var name.
    """
    import re as _re

    names: set[str] = set()

    raw = os.getenv("SHELL_ENV_EXCLUDE", "")
    names.update(v.strip() for v in raw.split(",") if v.strip())

    from_path = os.getenv("SHELL_ENV_EXCLUDE_FROM", "")
    if from_path:
        path = os.path.expanduser(from_path)
        if os.path.isfile(path):
            pattern = _re.compile(r"^export\s+([A-Za-z_][A-Za-z_0-9]*)\s*=")
            with open(path) as fh:
                for line in fh:
                    m = pattern.match(line.strip())
                    if m:
                        names.add(m.group(1))

    return frozenset(names)


# Env var names that are secrets by shape. `SHELL_PASS_ENV=1` hands the
# backend's environment to a subprocess the *model* drives, so the default has
# to be "don't", not "whatever the operator remembered to list". Matched
# case-insensitively against the whole name; `SHELL_ENV_ALLOW` puts one back.
# Substring globs, not suffix ones: `PGPASSWORD` has no underscore, so
# `*_PASSWORD` misses it.
_SECRET_NAME_GLOBS: tuple[str, ...] = (
    "*KEY",           # *_API_KEY, SECRET_KEY, bare OPENAI_KEY / SUPABASE_KEY
    "*APIKEY*",
    "*TOKEN*",
    "*SECRET*",
    "*PASSWORD*",
    "*PASSWD*",
    "*_PWD",          # MYSQL_PWD
    "*CREDENTIAL*",
    "*_AUTH",
    "*_DSN",          # SENTRY_DSN embeds the project key
    "AWS_*",
    "AZURE_*",
    "GOOGLE_*",
    "GCP_*",
    # Connection strings carry inline credentials. Narrow to the schemes that
    # actually do, rather than every *_URL — LANGFUSE_BASE_URL and friends are
    # addresses, not secrets, and over-blocking sends people to SHELL_ENV_ALLOW
    # for no benefit.
    "DATABASE_URL",
    "*POSTGRES*_URL",
    "*POSTGRESQL*_URL",
    "*MYSQL*_URL",
    "*REDIS*_URL",
    "*MONGO*_URI",
    "*AMQP*_URL",
    "*_CONNECTION_STRING",
)

# Names that are neither secret-shaped nor secrets, but hand the subprocess
# the *use* of one.
_SECRET_NAME_EXACT: frozenset[str] = frozenset({
    # A live agent socket: not a secret string, but signing with the
    # operator's SSH keys is one `ssh` away.
    "SSH_AUTH_SOCK",
    # Cluster credentials, or a path to them.
    "KUBECONFIG",
    # Registry auth blob.
    "DOCKER_AUTH_CONFIG",
    # Path to a service-account JSON key.
    "GOOGLE_APPLICATION_CREDENTIALS",
    "PGPASSFILE",
    "NETRC",
})


def is_secret_env_name(name: str) -> bool:
    """Whether an env var name looks like a credential, or grants use of one.

    Note the limit of this whole approach: `HOME` is deliberately forwarded so
    startup scripts work, which leaves `~/.aws/credentials`, `~/.ssh/id_rsa`
    and `~/.netrc` one `cat` away. This closes the environment channel, not
    the filesystem one — `SHELL_TOOL_WORKSPACE` and not enabling the shell
    tool on a host that holds credentials are what cover that.
    """
    upper = name.upper()
    if upper in _SECRET_NAME_EXACT:
        return True
    return any(fnmatch(upper, glob) for glob in _SECRET_NAME_GLOBS)


def _shell_env_allow() -> frozenset[str]:
    """`SHELL_ENV_ALLOW`: names to forward even though they look secret.

    For the startup script that genuinely needs one credential (a `gh auth`
    wrapper, say) — name it, rather than dropping the whole default.
    """
    raw = os.getenv("SHELL_ENV_ALLOW", "")
    return frozenset(v.strip() for v in raw.split(",") if v.strip())


def shell_env_filter() -> Callable[[str], bool]:
    """`shell_env_excluded`, bound to one snapshot of the rules.

    Both rule sets are read from the environment on every call, and
    `SHELL_ENV_EXCLUDE_FROM` re-opens and re-scans a file to build one of them.
    Filtering a whole environment calls the predicate once per variable, so
    take the snapshot first.
    """
    allow = _shell_env_allow()
    exclude = _shell_env_exclude()

    def excluded(name: str) -> bool:
        if name in allow:
            return False
        return name in exclude or is_secret_env_name(name)

    return excluded


def shell_env_excluded(name: str) -> bool:
    """Whether `name` is withheld from the shell subprocess. For one-off
    questions — to filter a whole environment, use `shell_env_filter`."""
    return shell_env_filter()(name)


def _shell_subprocess_env() -> dict[str, str] | None:
    """Environment for the shell subprocess, or None to inherit nothing.

    `None` is the default and is *not* the same as "inherit": ShellToolMiddleware
    passes `env={}` to Popen in that case, so the subprocess gets no PATH/HOME
    either. `SHELL_PASS_ENV=1` opts into a filtered copy.
    """
    if not os.getenv("SHELL_PASS_ENV"):
        return None
    excluded = shell_env_filter()
    return {k: v for k, v in os.environ.items() if not excluded(k)}


def _build_startup_commands() -> list[str]:
    """Load shell startup commands from a local script file.

    Reads ``SHELL_STARTUP_SCRIPT`` env var, falling back to
    ``shell_startup.local.sh`` next to this file (gitignored).
    Non-empty, non-comment lines become startup commands for the shell session.
    See ``shell_startup.example.sh`` for patterns (gh auth, aliases, etc.).
    """
    script_path = os.getenv("SHELL_STARTUP_SCRIPT", _SHELL_STARTUP_DEFAULT)
    if not os.path.isfile(script_path):
        return []
    with open(script_path) as fh:
        return [
            line.rstrip("\n")
            for line in fh
            if line.strip() and not line.startswith("#")
        ]


def _build_shell_middlewares() -> list:
    """Return [ShellApprovalMiddleware, ShellToolMiddleware] for the shell spec.

    Approval is **always installed** alongside the shell tool — running shell
    commands unattended on the backend host is dangerous, so the safe default
    is to ask the user before every execution.  The client can opt out per-turn
    by setting ``state["shell_approval_disabled"] = True`` (driven by the iOS
    Settings sheet); the middleware honours that flag and skips the interrupt.

    Set ``SHELL_PASS_ENV=1`` to forward the backend's environment to the shell
    subprocess — minus anything ``shell_env_excluded`` withholds: the
    operator's ``SHELL_ENV_EXCLUDE`` list plus every secret-shaped name
    (``*_API_KEY``, ``AWS_*``, …). ``SHELL_ENV_ALLOW`` names exceptions.
    Without it the subprocess gets
    no inherited env — note that ``ShellToolMiddleware`` passes ``env={}`` to
    ``Popen`` when env is ``None``, which strips HOME and PATH, so startup
    commands that rely on those (e.g. the gh auth wrapper) require
    ``SHELL_PASS_ENV=1``.
    """
    from pupa_backend.harnesses.langgraph.shell_approval import ShellApprovalMiddleware

    workspace = os.getenv("SHELL_TOOL_WORKSPACE")
    env = _shell_subprocess_env()
    startup = _build_startup_commands()
    kwargs: dict[str, Any] = {"startup_commands": startup}
    if workspace:
        kwargs["workspace_root"] = workspace
    if env is not None:
        kwargs["env"] = env
    shell_mw = ShellToolMiddleware(**kwargs)
    return [ShellApprovalMiddleware(), shell_mw]


def _build_subagent_middleware(model: Any, tools: List[Any]) -> list:
    """Return [SubAgentMiddleware] wired with a StateBackend.

    Receives the main agent's ``model`` and ``tools`` so subagents can share
    the same LLM and tool set without re-importing agent.py (which would
    create a circular dependency).

    The middleware adds a ``task`` tool to the main agent that lets it
    delegate isolated sub-tasks to specialist subagents.  Each subagent runs
    ephemerally and returns a single result.  Extend the ``subagents`` list
    to add more specialist agents.

    ``StateBackend()`` is used so subagents share the same LangGraph state
    as the main agent — no extra storage needed.
    """
    subagents: list[SubAgent] = [
        SubAgent(
            name="researcher",
            description=(
                "Performs focused research tasks: web searches, fact-checking, "
                "and synthesising information into concise reports. "
                "Use when the main agent needs external data without polluting "
                "the orchestrator context."
            ),
            system_prompt=(
                "You are a research assistant. Your job is to gather, verify, "
                "and synthesise information on the topic you are given. "
                "Return a concise, structured report with key facts and sources. "
                "Do not speculate beyond what you find."
            ),
            model=model,
            tools=tools,
        ),
    ]
    task_description = (
        "Launch an ephemeral subagent to handle a complex, multi-step task in an isolated context window.\n\n"
        "Use this to offload heavy or focused work so it does not crowd your own conversation context. "
        "The subagent runs autonomously and returns a single result; you never see its intermediate steps.\n\n"
        "Available agent types:\n{available_agents}\n\n"
        "Usage:\n"
        "- Launch multiple agents in parallel (single message, multiple tool calls) when tasks are independent.\n"
        "- Write a fully self-contained prompt: the subagent has no access to this conversation history.\n"
        "- Skip this tool for trivial tasks (a few lookups) — delegate only when isolation or deep reasoning adds value.\n"
        "- The subagent's output is not shown to the user automatically; summarize it yourself."
        "- Useful to offload tasks that might be also simple but long and would crowd your conversation history (e.g. a multi-step reasoning chain, or a long web search with many results)."
    )
    return [SubAgentMiddleware(backend=StateBackend(), subagents=subagents, task_description=task_description)]


def _build_skills_middlewares() -> list:
    """Return the skill-lifecycle middleware (progressive disclosure over the
    built-in `skills_lib/` directory). Thin delegate to `skills.py` so the
    deepagents `SkillsMiddleware` wiring lives next to its `read_skill` tool."""
    from pupa_backend.harnesses.langgraph.skills import build_skills_middlewares

    return build_skills_middlewares()


def _build_claude_code_tool() -> Any:
    """Return the `claude_code` tool. Thin delegate to `claude_code_tool.py` so
    the subprocess logic lives next to its tool definition (lazy import keeps
    the CLI shell-out off the import path until the tool is actually enabled)."""
    from pupa_backend.harnesses.langgraph.claude_code_tool import build_claude_code_tool

    return build_claude_code_tool()


BACKEND_TOOLS: List[BackendToolSpec] = [
    BackendToolSpec(
        name="tavily_search",
        description=(
            "Web search via Tavily — lets the agent look up real-world facts "
            "(prices, current events, product specs) mid-turn instead of "
            "guessing from training data."
        ),
        env_var="TAVILY_API_KEY",
        factory=lambda: TavilySearch(max_results=5),
    ),
    BackendToolSpec(
        name="shell",
        description=(
            "Local shell — lets the agent run shell commands on the backend "
            "host (read files, run tests, check git log). Full host access; "
            "intended for trusted dev environments only."
        ),
        env_var="SHELL_TOOL_ENABLED",
        middleware_factory=_build_shell_middlewares,
    ),
    BackendToolSpec(
        name="subagents",
        description=(
            "SubAgentMiddleware — adds a `task` tool that lets the main agent "
            "delegate isolated sub-tasks to specialist subagents (e.g. researcher). "
            "On by default; opt out with PUPA_SUBAGENTS_DISABLED=1."
        ),
        env_var=None,  # default ON
        disable_env_var="PUPA_SUBAGENTS_DISABLED",
        middleware_factory=_build_subagent_middleware,
        gated_tool_names=("task",),
    ),
    BackendToolSpec(
        name="skills",
        description=(
            "Agent skills — SKILL.md workflows under ~/.pupa-backend/skills/ "
            "surfaced via progressive disclosure: names + descriptions injected "
            "at agent start, full bodies loaded on demand through the read-only "
            "`skill_view` tool. On by default; opt out with PUPA_SKILLS_DISABLED=1."
        ),
        env_var=None,  # default ON
        disable_env_var="PUPA_SKILLS_DISABLED",
        middleware_factory=_build_skills_middlewares,
    ),
    BackendToolSpec(
        name="claude_code",
        description=(
            "Delegate a self-contained coding/research job to a full Claude "
            "Code agent (claude -p) running in its own context and tool loop. "
            "Returns a single synthesized result. Read-only by default. On by "
            "default; opt out with PUPA_CLAUDE_CODE_DISABLED=1."
        ),
        env_var=None,  # default ON
        disable_env_var="PUPA_CLAUDE_CODE_DISABLED",
        factory=_build_claude_code_tool,
    ),
]


def mcp_server_specs() -> List[BackendToolSpec]:
    """One informational spec per configured MCP server (`PUPA_MCP_SERVERS`).

    Discovery-only (no factory/middleware — the tools come from
    `MCPServersLifecycle` in app.py) so the iOS Settings sheet reflects which
    config-driven MCP servers the backend has wired up. A server with
    `enabled: false` is omitted. Names are prefixed `mcp_` to avoid colliding with
    the static registry. Playwright is an ordinary entry here (no longer special-
    cased). The server's optional `description:` is surfaced verbatim; otherwise a
    generic line is used. Tools are gated behind `get_tools(server=...)` at runtime.
    """
    import json as _json

    raw = os.getenv("PUPA_MCP_SERVERS")
    if not raw:
        return []
    try:
        block = _json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(block, dict):
        return []

    specs: List[BackendToolSpec] = []
    for name, entry in block.items():
        if isinstance(entry, dict) and entry.get("enabled") is False:
            continue
        desc = entry.get("description") if isinstance(entry, dict) else None
        if not (isinstance(desc, str) and desc.strip()):
            desc = (
                f"MCP server '{name}' — tools loaded from the config-driven "
                "mcp_servers block, gated behind get_tools."
            )
        specs.append(
            BackendToolSpec(
                name=f"mcp_{name}",
                description=desc.strip(),
                env_var=None,  # presence in PUPA_MCP_SERVERS is the gate
            )
        )
    return specs


def static_tool_aliases() -> dict[str, set[str]]:
    """Map each static spec's disable id → the real tool name(s) it gates.

    Only specs that register tools under a different name (e.g. `subagents` →
    `task`) appear. `ToolGatingMiddleware` uses this to translate a client mute
    into the names actually present in `request.tools`. MCP aliases are added
    separately by `build_graph` from the live `MCPServersLifecycle`.
    """
    return {
        spec.name: set(spec.gated_tool_names)
        for spec in BACKEND_TOOLS
        if spec.gated_tool_names
    }


def all_specs() -> List[BackendToolSpec]:
    """Static registry plus the dynamic, config-driven MCP server specs."""
    return BACKEND_TOOLS + mcp_server_specs()


def enabled_specs() -> List[BackendToolSpec]:
    """Specs whose env-var gate is currently satisfied."""
    return [spec for spec in BACKEND_TOOLS if spec.enabled_by_env]


def build_tools() -> List[Any]:
    """Materialise every env-enabled regular tool. Called once at graph build time."""
    return [spec.factory() for spec in enabled_specs() if spec.factory is not None]


def build_middlewares(model: Any = None, tools: List[Any] | None = None) -> List[Any]:
    """Materialise every env-enabled tool-providing middleware.

    A factory may return a single middleware or a list of middlewares (e.g.
    the shell spec returns [ShellApprovalMiddleware, ShellToolMiddleware] when
    approval is enabled).  Lists are flattened so callers always get a flat
    sequence suitable for `create_agent(middleware=[...])`.

    Args:
        model: The main agent's LLM, forwarded to middleware factories that
            need it (e.g. SubAgentMiddleware so subagents share the same model).
        tools: The main agent's tools, forwarded likewise.
    """
    import inspect as _inspect
    result: List[Any] = []
    for spec in enabled_specs():
        if spec.middleware_factory is None:
            continue
        _params = _inspect.signature(spec.middleware_factory).parameters
        if "model" in _params:
            produced = spec.middleware_factory(model=model, tools=tools or [])
        else:
            produced = spec.middleware_factory()
        if isinstance(produced, list):
            result.extend(produced)
        else:
            result.append(produced)
    return result
