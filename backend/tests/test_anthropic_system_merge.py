"""Pin the system-message normalisation used by `SystemMergingChatAnthropic`.

`ChatAnthropic` raises `ValueError: Received multiple non-consecutive system
messages`; `ChatBedrockConverse` tolerates the same shape. The merge helper
exists to mirror the Converse behaviour for the Anthropic path so middlewares
that append a separate `SystemMessage` (e.g. `TodoListMiddleware`'s planning
fragment) don't break the agent.
"""



from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from pupa_backend.harnesses.langgraph.anthropic_client import merge_non_consecutive_system_messages


def test_no_system_messages_returns_input_unchanged() -> None:
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    out = merge_non_consecutive_system_messages(msgs)
    assert [type(m) for m in out] == [HumanMessage, AIMessage]
    assert out[0].content == "hi"
    assert out[1].content == "hello"


def test_single_leading_system_message_passthrough() -> None:
    msgs = [SystemMessage("S1"), HumanMessage("hi"), AIMessage("hello")]
    out = merge_non_consecutive_system_messages(msgs)
    assert [type(m) for m in out] == [SystemMessage, HumanMessage, AIMessage]
    assert out[0].content == "S1"


def test_two_non_consecutive_system_messages_merge() -> None:
    msgs = [
        SystemMessage("S1"),
        HumanMessage("hi"),
        SystemMessage("S2"),
        AIMessage("hello"),
    ]
    out = merge_non_consecutive_system_messages(msgs)

    assert [type(m) for m in out] == [SystemMessage, HumanMessage, AIMessage]
    assert out[0].content == "S1\n\nS2"
    assert out[1].content == "hi"
    assert out[2].content == "hello"


def test_three_systems_interleaved_with_tools_preserve_order() -> None:
    msgs = [
        SystemMessage("base"),
        HumanMessage("ask"),
        SystemMessage("planning fragment"),
        AIMessage("call tool"),
        ToolMessage(content="tool result", tool_call_id="tc1"),
        SystemMessage("late inject"),
        AIMessage("final"),
    ]
    out = merge_non_consecutive_system_messages(msgs)

    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "base\n\nplanning fragment\n\nlate inject"

    non_system = out[1:]
    assert [type(m) for m in non_system] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert [m.content for m in non_system] == [
        "ask",
        "call tool",
        "tool result",
        "final",
    ]


def test_empty_system_content_is_skipped() -> None:
    msgs = [
        SystemMessage(""),
        HumanMessage("hi"),
        SystemMessage("S2"),
    ]
    out = merge_non_consecutive_system_messages(msgs)
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "S2"


def test_list_block_system_content_concatenates() -> None:
    msgs = [
        SystemMessage(content=[{"type": "text", "text": "block-a"}]),
        HumanMessage("hi"),
        SystemMessage(content=[{"type": "text", "text": "block-b"}]),
    ]
    out = merge_non_consecutive_system_messages(msgs)
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == [
        {"type": "text", "text": "block-a"},
        {"type": "text", "text": "block-b"},
    ]


def test_mixed_str_and_list_block_system_content() -> None:
    msgs = [
        SystemMessage("S1"),
        HumanMessage("hi"),
        SystemMessage(content=[{"type": "text", "text": "block-b"}]),
    ]
    out = merge_non_consecutive_system_messages(msgs)
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == [
        {"type": "text", "text": "S1"},
        {"type": "text", "text": "block-b"},
    ]
