"""Claude Code agent harness — Claude Agent SDK drives the frontend tools.

Enabled via the config.yml `harnesses:` block (see `harnesses.py`) and mounted at
`POST /harnesses/claude_code` (and `/` if it is the default harness). With it
active, `claude-agent-sdk` runs in-process as the tool-calling loop and the
iOS-forwarded frontend tools become in-process SDK MCP tools. When Claude calls
one, the loop round-trips to iOS over the existing AG-UI interrupt/resume
contract — **iOS is unchanged**.

## Confirmed `claude-agent-sdk` API (v0.2.106, verified against the install)

- **Wraps the `claude` CLI** — `ClaudeSDKClient` spawns the `claude` binary
  (`shutil.which("claude")` / known paths) over stdio, so it inherits Claude
  Code's auth/billing resolution. This is what makes subscription billing
  possible (and what `env.py` must defend — see below).
- **Subprocess env**: the transport builds the child env as
  ``{**os.environ (minus CLAUDECODE), **options.env}``. `options.env` is an
  **overlay** — it cannot *delete* a var already in the parent env. There is no
  "don't inherit" switch. ⇒ enforcement of subscription-only billing is
  **detect-forbidden-var-and-refuse**, not silent scrub (`env.assert_*`).
- `ClaudeAgentOptions(system_prompt, mcp_servers={name: cfg}, allowed_tools,
  disallowed_tools, can_use_tool, env, permission_mode, cwd, model, resume,
  include_partial_messages, ...)`.
- `@tool(name, description, input_schema)` decorator → `SdkMcpTool`; handler is
  `async (args: dict) -> dict` returning an MCP result
  ``{"content": [{"type": "text", "text": ...}]}``. The handler does **not**
  receive the tool_use id (see `registry` for the `(name, args)` correlation).
- `create_sdk_mcp_server(name, version="1.0.0", tools=[...])` → in-process
  `McpSdkServerConfig`. Claude sees tools as ``mcp__<server>__<tool>``.
- `can_use_tool`: `async (tool_name: str, input: dict, ctx: ToolPermissionContext)
  -> PermissionResultAllow | PermissionResultDeny`. Allow defaults to
  `behavior="allow"`; Deny takes `message` + `interrupt`.
- Driving: `client = ClaudeSDKClient(options)`; `await client.connect()`;
  `await client.query(prompt)`; `async for msg in client.receive_response()`
  yields `AssistantMessage` (`.content` = list of `TextBlock` / `ToolUseBlock`
  / `ThinkingBlock`; `.message_id`, `.session_id`), then a terminal
  `ResultMessage` (`.session_id`, `.total_cost_usd`, `.is_error`, `.result`).
  With `include_partial_messages=True` (the loop sets it), `receive_response()`
  also yields partial `StreamEvent`s (`.event` = the raw Anthropic API stream
  event) so assistant text streams token-by-token; the assembled
  `AssistantMessage` still follows. `await client.disconnect()` tears the
  subprocess down.
- **Auth probe**: `claude auth status --json` →
  ``{loggedIn, authMethod, apiProvider, apiKeySource?}``. Subscription methods
  are `authMethod ∈ {"claude.ai", "oauth_token"}` with `apiProvider=="firstParty"`;
  `api_key` / `third_party` / `none` / unknown ⇒ refuse (`env.py`).
"""

from __future__ import annotations

from .endpoint import register_claude_loop_endpoint
from .env import SubscriptionBillingUnavailable

__all__ = ["register_claude_loop_endpoint", "SubscriptionBillingUnavailable"]
