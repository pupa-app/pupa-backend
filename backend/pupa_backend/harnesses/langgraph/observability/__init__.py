"""Langfuse tracing and usage reporting.

Named for the role rather than the vendor: a package called `langfuse` sitting
next to `from langfuse.langchain import CallbackHandler` reads as a shadowing
bug even though absolute imports resolve it correctly.

    tracing.py — the *write* path. Stamps every run with a release + trace id
                 and builds the LangChain callback handler.
    usage.py   — the *read* path. Pulls token/cache usage back out of Langfuse
                 for `/db/threads/usage`.
"""

from .tracing import (
    backend_version,
    langfuse_enabled,
    langfuse_envs_present,
    resolve_langfuse_config,
)
from .usage import fetch_cache, fetch_usage

__all__ = [
    "backend_version",
    "fetch_cache",
    "fetch_usage",
    "langfuse_enabled",
    "langfuse_envs_present",
    "resolve_langfuse_config",
]
