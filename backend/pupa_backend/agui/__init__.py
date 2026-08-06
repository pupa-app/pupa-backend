"""AG-UI wire-protocol helpers shared by every harness.

Nothing in here may import from `pupa_backend.harnesses` — these are the
pieces both the LangGraph and Claude Code loops need to speak AG-UI, so
they live above the harness boundary rather than inside either one.
"""

from .tool_results import parse_tool_results

__all__ = ["parse_tool_results"]
