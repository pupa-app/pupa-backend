"""Shared test fixtures and the MockChatModel used by the spiral regression suite.

`MockChatModel` implements `_stream` / `_astream` over a queue of canned
`AIMessage` responses, so tests can drive `create_agent` end-to-end through
real `awrap_model_call` / `_fix_messages_for_bedrock` paths without touching
Bedrock.

This module also injects dummy AWS credentials (via `setdefault`) so tests
that import `agent.py` — which builds a `ChatBedrockConverse` client at
module load — succeed without real Bedrock creds. Live creds in the env
override the dummies; CI and laptops without any AWS setup still run the
suite.
"""



import json
import os
from typing import Any

os.environ.setdefault("LLM_PROVIDER", "bedrock")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


class MockChatModel(BaseChatModel):
    """Mock LLM that yields pre-configured `AIMessage`s in order.

    `responses` is a list of either single `AIMessage`s or lists of them
    (one list = one stream of chunks). Implements both sync `_stream` and
    `_generate`.
    """

    responses: list[Any] = Field(default_factory=list)
    _call_count: int = PrivateAttr(default=0)
    _call_args_list: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def stream_call_args_list(self) -> list:
        return self._call_args_list

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ARG002
        self._call_args_list.append(list(messages))
        response = self.responses[self._call_count]
        self._call_count += 1
        final = response[-1] if isinstance(response, list) else response
        return ChatResult(generations=[ChatGeneration(message=final)])

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        self._call_args_list.append(list(messages))
        response = self.responses[self._call_count]
        self._call_count += 1
        chunks = response if isinstance(response, list) else [response]
        for msg in chunks:
            tool_call_chunks = []
            for call_chunk in (msg.tool_calls or []):
                args = call_chunk["args"]
                tool_call_chunks.append(ToolCallChunk(
                    name=call_chunk["name"],
                    args=json.dumps(args) if isinstance(args, dict) else args,
                    id=call_chunk["id"],
                    index=0,
                ))
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=msg.content,
                    tool_call_chunks=tool_call_chunks,
                    id=getattr(msg, "id", None),
                )
            )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MockChatModel":  # noqa: ARG002
        return self
