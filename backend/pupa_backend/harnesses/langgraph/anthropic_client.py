"""ChatAnthropic wrapper that tolerates non-consecutive `SystemMessage`s.

`ChatAnthropic._format_messages` raises `ValueError: Received multiple
non-consecutive system messages` when the message list contains more than
one `SystemMessage` separated by other roles. `ChatBedrockConverse` silently
collapses them, so the rest of the agent graph (notably `TodoListMiddleware`,
which appends a planning system fragment on top of the agent's base system
prompt) works against Bedrock but blows up against the direct Anthropic API.

This module mirrors the Converse behaviour by merging every `SystemMessage`
in the input into one leading entry before delegating to `ChatAnthropic`.
"""



from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGenerationChunk


def merge_non_consecutive_system_messages(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Collapse every `SystemMessage` in `messages` into one leading entry.

    Non-`SystemMessage` entries keep their relative order. If the input has
    no `SystemMessage`, the list is returned as-is (new list, same items).
    String contents are joined with a blank line; list-of-blocks contents
    are concatenated.
    """
    system_text_parts: list[str] = []
    system_block_parts: list[Any] = []
    rest: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            if isinstance(msg.content, str):
                if msg.content:
                    system_text_parts.append(msg.content)
            else:
                system_block_parts.extend(msg.content)
        else:
            rest.append(msg)

    if not system_text_parts and not system_block_parts:
        return list(messages)

    if system_block_parts and not system_text_parts:
        merged_content: Any = system_block_parts
    elif system_text_parts and not system_block_parts:
        merged_content = "\n\n".join(system_text_parts)
    else:
        merged_content = [
            *( [{"type": "text", "text": "\n\n".join(system_text_parts)}]
               if system_text_parts else [] ),
            *system_block_parts,
        ]

    return [SystemMessage(content=merged_content), *rest]


class SystemMergingChatAnthropic(ChatAnthropic):
    """`ChatAnthropic` that pre-merges non-consecutive `SystemMessage`s.

    The Anthropic SDK rejects a message list with more than one
    `SystemMessage` when they're not adjacent. `ChatBedrockConverse`
    tolerates the same shape, so middlewares that append a separate
    system fragment (e.g. `TodoListMiddleware`'s planning prompt) work on
    Bedrock but fail against `ChatAnthropic`. Normalising up front mirrors
    Converse's behaviour without touching the rest of the agent graph.
    """

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return super()._generate(
            merge_non_consecutive_system_messages(messages), *args, **kwargs
        )

    async def _agenerate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        return await super()._agenerate(
            merge_non_consecutive_system_messages(messages), *args, **kwargs
        )

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        yield from super()._stream(
            merge_non_consecutive_system_messages(messages), *args, **kwargs
        )

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in super()._astream(
            merge_non_consecutive_system_messages(messages), *args, **kwargs
        ):
            yield chunk
