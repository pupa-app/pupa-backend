"""Pupa agent — Bedrock Claude + CustomCopilotKitMiddleware.

The canvas is a native iOS / macOS app (the web frontend is unsupported).
The agent shapes the canvas by calling FRONTEND TOOLS registered by the
client; their names, JSON Schema parameters, and short descriptions are
forwarded to the model as proper tool definitions. **Don't enumerate
those tools in the system prompt** — duplicating descriptions causes drift
when the client surface changes. The prompt below only sets behavioural
rules (style, anti-duplication, mid-turn state freshness, flow).

Frontend-tool dispatch is interrupt-driven: `CustomCopilotKitMiddleware`
batches every frontend tool_call the model emits into one
`langgraph.interrupt(...)`, the iOS client executes the calls locally and
POSTs the per-call results back via `forwardedProps.command.resume`, and
the middleware appends one `ToolMessage` per result before the model is
re-invoked. The model never speaks past a pending frontend tool_call
without first seeing its result — the same guarantee Anthropic's tool_use
loop provides, enforced structurally rather than by prompt instruction.

Backend tools (currently `tavily_search`, the env-gated `shell` registered
by `ShellToolMiddleware`, and `write_todos`) live in `backend_tools.py` and
`TodoListMiddleware` respectively. `ToolGatingMiddleware` then lets the iOS
Settings sheet mute any of them per turn via
`RunAgentInput.state["disabled_tools"]`.

`LG_RECURSION_LIMIT` (default 100) bumps LangGraph's per-run step cap
"""

import logging
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, TodoListMiddleware, before_agent
from langchain_aws import ChatBedrockConverse
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.config import get_config
from langgraph.runtime import Runtime

from pupa_backend.harnesses.langgraph.backend_tools import build_middlewares, build_tools, static_tool_aliases
from pupa_backend.credentials import get_credential
from pupa_backend.harnesses.langgraph.frontend_interrupt import CustomCopilotKitMiddleware
from pupa_backend.prompts import SYSTEM_PROMPT
from pupa_backend.harnesses.langgraph.tool_gating import ToolGatingMiddleware

logger = logging.getLogger("uvicorn.error")


# Trimmed `write_todos` tool description. langchain's default ships a
# ~3.6k-char prose block (~913 tokens) — the longest single contributor
# in our per-turn payload. This version preserves the
# behavioural rules (when to use, state values, mark-completed-immediately,
# no parallel calls) and drops examples + redundant prose.
WRITE_TODOS_TOOL_DESCRIPTION = (
    "Track a plan for the current session as a list of todos "
    "({content, activeForm, status: pending|in_progress|completed}). "
    "Use only for multi-step work (3+ distinct steps); skip for trivial "
    "or single-step requests. Mark a todo `in_progress` BEFORE working it, "
    "`completed` IMMEDIATELY after finishing — never batch completions. "
    "Keep at least one todo `in_progress` while work remains. Never call "
    "this tool in parallel; revise the list as new info arrives."
)

# Trimmed `TodoListMiddleware` system-prompt fragment. langchain's default
# is ~1k chars (~268 tokens); the rules below are also restated in the
# tool description above, so this fragment only needs to point the model
# at the tool and gate it to non-trivial work.
WRITE_TODOS_SYSTEM_PROMPT = (
    "Use the `write_todos` tool to plan multi-step work and surface progress "
    "to the user. Skip it for trivial / single-step requests. Mark items "
    "completed as soon as the step is done; don't batch."
)


@before_agent
def _log_thread_id(state: dict[str, Any], runtime: Runtime[Any]) -> None:
    """Log the LangGraph thread_id at the start of each agent turn.

    Useful for correlating backend logs with a specific chat session — the
    iOS client mints one ``threadId`` per New Session and reuses it for
    every subsequent turn.
    """
    thread_id = get_config().get("configurable", {}).get("thread_id")
    logger.info("[pupa] thread_id=%r", thread_id)

PROVIDER_BEDROCK = "bedrock"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_OPENROUTER = "openrouter"

# OpenRouter is an OpenAI-compatible aggregator: one base_url + one key serves
# every model, so — unlike `openai_compatible`, which is env-pinned to a single
# `LLM_MODEL` — we register a curated menu of slugs here and let the client pick.
# `model_id` values are OpenRouter's canonical `id` strings; verify against
# https://openrouter.ai/models before adding new ones (a stale slug 404s at call
# time). Auth is the single `OPENROUTER_API_KEY` env var.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Logical model id → provider-specific construction kwargs, keyed by (provider, logical_id).
# iOS clients send the (provider, logical_id) pair in `forwardedProps["llm"]`; the backend
# maps them to provider-specific strings here so iOS stays ignorant of e.g. Bedrock's
# `eu.anthropic.…` inference profiles. Add new entries as new models ship.
MODEL_REGISTRY: dict[tuple[str, str], dict[str, Any]] = {
    # --- Anthropic (direct API) ---
    (PROVIDER_ANTHROPIC, "claude-opus-4-8"): {
        "model_id": "claude-opus-4-8",
        "label": "Claude Opus 4.8",
    },
    (PROVIDER_ANTHROPIC, "claude-sonnet-4-6"): {
        "model_id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
    },
    (PROVIDER_ANTHROPIC, "claude-haiku-4-5"): {
        "model_id": "claude-haiku-4-5",
        "label": "Claude Haiku 4.5",
    },
    # --- Bedrock (EU cross-region inference profiles) ---
    (PROVIDER_BEDROCK, "claude-opus-4-8"): {
        "model_id": "eu.anthropic.claude-opus-4-8",
        "region": "eu-west-1",
        "label": "Claude Opus 4.8",
    },
    (PROVIDER_BEDROCK, "claude-sonnet-4-6"): {
        "model_id": "eu.anthropic.claude-sonnet-4-6",
        "region": "eu-west-1",
        "label": "Claude Sonnet 4.6",
    },
    (PROVIDER_BEDROCK, "claude-haiku-4-5"): {
        "model_id": "eu.anthropic.claude-haiku-4-5",
        "region": "eu-west-1",
        "label": "Claude Haiku 4.5",
    },
    # --- OpenRouter (popular open / frontier models) ---
    (PROVIDER_OPENROUTER, "glm-5.1"): {
        "model_id": "z-ai/glm-5.1",
        "label": "GLM-5.1",
    },
    (PROVIDER_OPENROUTER, "qwen3.7-max"): {
        "model_id": "qwen/qwen3.7-max",
        "label": "Qwen3.7 Max",
    },
    (PROVIDER_OPENROUTER, "qwen3.6-plus"): {
        "model_id": "qwen/qwen3.6-plus",
        "label": "Qwen3.6 Plus",
    },
    (PROVIDER_OPENROUTER, "qwen3.6-35b-a3b"): {
        "model_id": "qwen/qwen3.6-35b-a3b",
        "label": "Qwen3.6 35B A3B",
    },
    (PROVIDER_OPENROUTER, "gpt-oss-120b:free"): {
        "model_id": "openai/gpt-oss-120b:free",
        "label": "GPT-OSS 120B Free",
    },
    (PROVIDER_OPENROUTER, "gpt-oss-20b:free"): {
        "model_id": "openai/gpt-oss-20b:free",
        "label": "GPT-OSS 20B Free",
    },
    (PROVIDER_OPENROUTER, "minimax-m3"): {
        "model_id": "minimax/minimax-m3",
        "label": "MiniMax M3",
    },
    (PROVIDER_OPENROUTER, "kimi-k2.6"): {
        "model_id": "moonshotai/kimi-k2.6",
        "label": "Kimi K2.6",
    },
    (PROVIDER_OPENROUTER, "deepseek-v4-pro"): {
        "model_id": "deepseek/deepseek-v4-pro",
        "label": "DeepSeek V4 Pro",
    },
    (PROVIDER_OPENROUTER, "deepseek-v4-flash"): {
        "model_id": "deepseek/deepseek-v4-flash",
        "label": "DeepSeek V4 Flash",
    },
    (PROVIDER_OPENROUTER, "qwen2.5-72b-instruct"): {
        "model_id": "qwen/qwen-2.5-72b-instruct",
        "label": "Qwen2.5 72B Instruct",
    },
    (PROVIDER_OPENROUTER, "deepseek-r1-distill-llama-70b"): {
        "model_id": "deepseek/deepseek-r1-distill-llama-70b",
        "label": "DeepSeek R1 Distill Llama 70B",
    },
    (PROVIDER_OPENROUTER, "llama-3.3-70b"): {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "label": "Llama 3.3 70B",
    },
    (PROVIDER_OPENROUTER, "qwen3-235b-a22b"): {
        "model_id": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B-A22B",
    },
    (PROVIDER_OPENROUTER, "nemotron-3-super-120b-a12b:free"): {
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "label": "Nemotron 3 Super 120B Free",
    },
    (PROVIDER_OPENROUTER, "nemotron-3-ultra-550b-a55b:free"): {
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "label": "Nemotron 3 Ultra 550B Free",
    },
    (PROVIDER_OPENROUTER, "nemotron-3-nano-30b-a3b:free"): {
        "model_id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "label": "Nemotron 3 Nano 30B Free",
    },
    (PROVIDER_OPENROUTER, "nemotron-nano-9b-v2:free"): {
        "model_id": "nvidia/nemotron-nano-9b-v2:free",
        "label": "Nemotron Nano 9B V2 Free",
    },
}


class UnknownModelError(ValueError):
    """Raised when (provider, model) isn't in `MODEL_REGISTRY`. Surfaced to the
    iOS client as an AG-UI error event so the user sees a clear toast."""


class MissingCredentialsError(RuntimeError):
    """Raised when the chosen provider's credentials aren't in the backend env.
    Surfaced to the iOS client as an AG-UI error event."""


def _build_bedrock(model_id: str, region: str) -> BaseChatModel:
    # Credentials come via `get_credential` (the stash), not `os.environ`
    # directly: when the Claude Code harness coexists, AWS_* is scrubbed out of
    # the env at startup so the `claude` subprocess can't inherit it, and lives
    # in the in-process stash instead. Pass creds explicitly so boto3 doesn't
    # rely on the (now empty) env chain.
    profile = get_credential("AWS_PROFILE")
    access_key = get_credential("AWS_ACCESS_KEY_ID")
    if not (profile or access_key):
        raise MissingCredentialsError(
            "Bedrock selected but no AWS credentials in the backend env. "
            "Set AWS_PROFILE (recommended) or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
        )
    kwargs: dict[str, Any] = {"model_id": model_id, "region_name": region}
    if profile:
        kwargs["credentials_profile_name"] = profile
    if access_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = get_credential("AWS_SECRET_ACCESS_KEY")
        session_token = get_credential("AWS_SESSION_TOKEN")
        if session_token:
            kwargs["aws_session_token"] = session_token
    return ChatBedrockConverse(**kwargs)


def _build_anthropic(model_id: str) -> BaseChatModel:
    # `get_credential` reads the stash when ANTHROPIC_API_KEY was scrubbed for
    # Claude Code harness coexistence, else live env. See credentials.py.
    api_key = get_credential("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingCredentialsError(
            "Anthropic API selected but ANTHROPIC_API_KEY is not set in the backend env."
        )
    from pupa_backend.harnesses.langgraph.anthropic_client import SystemMergingChatAnthropic

    return SystemMergingChatAnthropic(
        model=model_id,
        api_key=api_key,
        model_kwargs={"cache_control": {"type": "ephemeral"}},
    )


def _build_openai_compatible_from_env() -> BaseChatModel:
    """OpenAI-compatible provider is env-only — base_url / api_key / model all
    come from the backend's `.env`; there's no registry entry because every
    deployment points at a different proxy."""
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        raise MissingCredentialsError(
            "LLM_PROVIDER=openai_compatible but LLM_BASE_URL is not set."
        )
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise MissingCredentialsError(
            "LLM_PROVIDER=openai_compatible but LLM_API_KEY is not set."
        )
    model = os.getenv("LLM_MODEL")
    if not model:
        raise MissingCredentialsError(
            "LLM_PROVIDER=openai_compatible but LLM_MODEL is not set."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=base_url, api_key=api_key, model=model)


def _build_openrouter(model_id: str) -> BaseChatModel:
    """OpenRouter via its OpenAI-compatible endpoint. `model_id` is the OpenRouter
    slug (e.g. `z-ai/glm-5.1`); auth is the single `OPENROUTER_API_KEY` env var."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise MissingCredentialsError(
            "OpenRouter selected but OPENROUTER_API_KEY is not set in the backend env."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, model=model_id)


def build_model(provider: str | None = None, model_id: str | None = None) -> BaseChatModel:
    """Build a chat model for an explicit (provider, model_id) or fall back to env defaults.

    - Both args `None` → env-driven default (`LLM_PROVIDER` + provider-specific env vars).
    - Both args set    → look up `MODEL_REGISTRY[(provider, model_id)]` and build.
                         Raises `UnknownModelError` if absent, `MissingCredentialsError`
                         if the chosen provider lacks creds in the backend env.

    Mixed inputs (only one of the pair set) raise `UnknownModelError` to keep call sites honest.
    """
    if provider is None and model_id is None:
        return _build_default_from_env()

    if provider is None or model_id is None:
        raise UnknownModelError(
            f"provider and model must both be set or both omitted (got provider={provider!r}, model={model_id!r})."
        )

    if provider == PROVIDER_OPENAI_COMPATIBLE:
        # No registry mapping — OpenAI-compatible deployments are env-configured.
        return _build_openai_compatible_from_env()

    params = MODEL_REGISTRY.get((provider, model_id))
    if params is None:
        known = sorted(f"{p}/{m}" for (p, m) in MODEL_REGISTRY)
        raise UnknownModelError(
            f"Unknown (provider, model) = ({provider!r}, {model_id!r}). "
            f"Known combos: {known}."
        )

    if provider == PROVIDER_BEDROCK:
        return _build_bedrock(model_id=params["model_id"], region=params["region"])
    if provider == PROVIDER_ANTHROPIC:
        return _build_anthropic(model_id=params["model_id"])
    if provider == PROVIDER_OPENROUTER:
        return _build_openrouter(model_id=params["model_id"])
    raise UnknownModelError(f"Provider {provider!r} is not supported.")


def _build_default_from_env() -> BaseChatModel:
    """Env-driven model construction — preserves the pre-per-agent behaviour.

    Used at startup (to power the default graph) and as the fallback whenever an
    iOS client doesn't send `forwardedProps["llm"]`.
    """
    try:
        provider = os.environ["LLM_PROVIDER"].lower()
    except KeyError as exc:
        raise RuntimeError(
            "No LLM provider configured. Declare an `llm_providers` block in "
            "~/.pupa-backend/config.yml (the first entry is used as the default "
            "when `default_llm_provider` is omitted), or set LLM_PROVIDER in the "
            "environment. Expected one of: bedrock, anthropic, openai_compatible, "
            "openrouter."
        ) from exc
    logger.info("[pupa] default LLM provider: %s", provider)

    if provider == PROVIDER_BEDROCK:
        params = MODEL_REGISTRY[(PROVIDER_BEDROCK, "claude-sonnet-4-6")]
        return _build_bedrock(model_id=params["model_id"], region=params["region"])
    if provider == PROVIDER_ANTHROPIC:
        params = MODEL_REGISTRY[(PROVIDER_ANTHROPIC, "claude-sonnet-4-6")]
        return _build_anthropic(model_id=params["model_id"])
    if provider == PROVIDER_OPENAI_COMPATIBLE:
        return _build_openai_compatible_from_env()
    if provider == PROVIDER_OPENROUTER:
        model = os.getenv("LLM_MODEL")
        if not model:
            raise MissingCredentialsError(
                "LLM_PROVIDER=openrouter but LLM_MODEL is not set "
                "(the OpenRouter model slug, e.g. anthropic/claude-sonnet-4.6)."
            )
        return _build_openrouter(model_id=model)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        "Expected 'bedrock', 'anthropic', 'openai_compatible', or 'openrouter'."
    )


_MODEL_CACHE: dict[tuple[str | None, str | None], BaseChatModel] = {}


def get_model(provider: str | None = None, model_id: str | None = None) -> BaseChatModel:
    """Cached `build_model` — one instance per (provider, model_id) across requests."""
    key = (provider, model_id)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = build_model(provider=provider, model_id=model_id)
    _MODEL_CACHE[key] = model
    return model

DEFAULT_RECURSION_LIMIT = 500
DEFAULT_CLEAR_TOOL_USES_TRIGGER = 40_000


def recursion_limit() -> int:
    """Read `LG_RECURSION_LIMIT` env var (default `DEFAULT_RECURSION_LIMIT`).

    Raises `RuntimeError` on non-integer or non-positive values — config
    bugs should fail loud at startup, not silently fall back.
    """
    raw = os.getenv("LG_RECURSION_LIMIT")
    if raw is None:
        return DEFAULT_RECURSION_LIMIT
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"LG_RECURSION_LIMIT must be an integer, got {raw!r}."
        ) from exc
    if value < 1:
        raise RuntimeError(
            f"LG_RECURSION_LIMIT must be >= 1, got {value}."
        )
    return value


def clear_tool_uses_trigger() -> int:
    """Read `LG_CLEAR_TOOL_USES_TRIGGER` env var (default `DEFAULT_CLEAR_TOOL_USES_TRIGGER`).

    Raises `RuntimeError` on non-integer or non-positive values.
    """
    raw = os.getenv("LG_CLEAR_TOOL_USES_TRIGGER")
    if raw is None:
        return DEFAULT_CLEAR_TOOL_USES_TRIGGER
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"LG_CLEAR_TOOL_USES_TRIGGER must be an integer, got {raw!r}."
        ) from exc
    if value < 1:
        raise RuntimeError(
            f"LG_CLEAR_TOOL_USES_TRIGGER must be >= 1, got {value}."
        )
    return value


def _build_cache_middleware(model: BaseChatModel) -> list:
    if isinstance(model, ChatBedrockConverse):
        return [BedrockPromptCachingMiddleware()]
    return []


def build_graph(checkpointer=None, store=None, mcp=None, model: BaseChatModel | None = None):
    """Build the pupa agent graph with the given persistence layer.

    The checkpointer and store are sourced from the FastAPI lifespan in
    `app.py`, which opens them from `DATABASE_URL` (or falls back to local
    SQLite).

    mcp: optional `MCPServersLifecycle` holding tools loaded from the
    config-driven `mcp_servers:` block (playwright included). When provided
    (and non-empty), every server's tools are injected behind a single
    `get_tools(server=...)` gate tool; an `McpGateMiddleware` keeps each
    server's tools hidden until the agent activates that server for the thread.

    model: explicit chat model to bind. When omitted, falls back to the env-driven
    default (`get_model(None, None)`). Callers in the per-request path
    (`get_graph`) always pass an explicit model so the swap is deterministic.
    """
    if model is None:
        model = get_model()
    tools = build_tools()

    extra_tools: list = []
    extra_middlewares: list = []
    # Disable-id → real tool name(s) for the per-turn Settings gate. Static
    # specs come from the registry; MCP servers map `mcp_<server>` → that
    # server's tool names so muting a server in Settings drops its tools.
    tool_aliases = static_tool_aliases()
    if mcp and mcp.tools:
        extra_tools.extend(mcp.tools)
        extra_tools.append(mcp.build_gate_tool())
        extra_middlewares.append(mcp.build_gate_middleware())
        for server, names in mcp.server_tool_names.items():
            tool_aliases[f"mcp_{server}"] = set(names)

    return create_agent(
        model=model,
        tools=tools + extra_tools,
        middleware=[
            _log_thread_id,
            *_build_cache_middleware(model),
            CustomCopilotKitMiddleware(),
            TodoListMiddleware(
                system_prompt=WRITE_TODOS_SYSTEM_PROMPT,
                tool_description=WRITE_TODOS_TOOL_DESCRIPTION,
            ),
            ToolGatingMiddleware(aliases=tool_aliases),
            *extra_middlewares,
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=clear_tool_uses_trigger(),
                        keep=3,           # keep the 3 most recent tool uses
                        clear_tool_inputs=False,  # preserve inputs for traceability
                    ),
                ],
            ),
            *build_middlewares(model=model, tools=tools),
        ],
        checkpointer=checkpointer,
        store=store,
        system_prompt=SYSTEM_PROMPT,
        name="pupa",
    )


# Per-(provider, model) graph cache. Populated lazily on first request for a given pair.
# The default graph (None, None) is pre-built and registered in the FastAPI lifespan so
# the very first request doesn't pay construction latency.
_GRAPH_CACHE: dict[tuple[str | None, str | None], Any] = {}
# Persistence handles stashed by the lifespan so the cache can build new graphs lazily.
# Treat as read-only after startup.
_graph_deps: dict[str, Any] = {}


def register_graph_deps(*, checkpointer, store, mcp=None) -> None:
    """Called once by the FastAPI lifespan to stash the shared persistence layer
    so `get_graph` can lazily build new (provider, model) graphs at request time."""
    _graph_deps["checkpointer"] = checkpointer
    _graph_deps["store"] = store
    _graph_deps["mcp"] = mcp


def get_graph(provider: str | None = None, model_id: str | None = None):
    """Return the cached graph for (provider, model_id), building it on first miss.

    Raises `UnknownModelError` / `MissingCredentialsError` from `get_model` —
    callers (the FastAPI handler) translate these into AG-UI error events.
    """
    key = (provider, model_id)
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    if not _graph_deps:
        raise RuntimeError(
            "get_graph called before register_graph_deps — the FastAPI lifespan "
            "must register persistence handles before requests are served."
        )
    model = get_model(provider=provider, model_id=model_id)
    graph = build_graph(
        checkpointer=_graph_deps["checkpointer"],
        store=_graph_deps["store"],
        mcp=_graph_deps.get("mcp"),
        model=model,
    )
    _GRAPH_CACHE[key] = graph
    return graph
