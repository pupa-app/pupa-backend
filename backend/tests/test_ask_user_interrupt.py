"""Regression suite for ``ask_user_questions`` as a frontend tool.

``ask_user_questions`` used to be a backend tool whose body called
``langgraph.types.interrupt(...)`` directly. With
``CustomCopilotKitMiddleware`` in place, it's just another frontend
tool — its descriptor is forwarded to the model via
``state["copilotkit"]["actions"]`` and the middleware handles the
pause/resume. The iOS host renders the questions panel inside the
tool's handler and returns the answers as the tool result.

These tests pin the wire-level contract by driving
``CustomCopilotKitMiddleware`` end-to-end:

1. **Single-question pause** — one question in the descriptor pauses
   the graph on a ``frontend_tool_calls`` interrupt carrying the
   ``ask_user_questions`` call.
2. **Single-question resume** — re-invoking with
   ``Command(resume={"tool_results": [...]})`` injects a ToolMessage
   whose content is the JSON-encoded answers list; the model continues
   and emits the final assistant text.
3. **Multi-question batch** — the model calls ``ask_user_questions``
   with three questions; they all show up on a single interrupt.
4. **Mixed with another frontend tool** — ``ask_user_questions``
   batched alongside a canvas mutator in the same assistant turn:
   both pause together and both results return in one resume.
"""



from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pupa_backend.harnesses.langgraph.frontend_interrupt import CustomCopilotKitMiddleware

from .conftest import MockChatModel


ASK_USER_DESCRIPTOR = {
    "name": "ask_user_questions",
    "description": "Ask the user one or more clarifying questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question"],
                },
            },
        },
        "required": ["questions"],
    },
}

ADD_ITEM_DESCRIPTOR = {
    "name": "addItem",
    "description": "append one item to the tracker",
    "parameters": {
        "type": "object",
        "properties": {"item": {"type": "object"}},
        "required": ["item"],
    },
}


def _build_graph(model: MockChatModel, checkpointer: MemorySaver):
    return create_agent(
        model=model,
        tools=[],
        middleware=[CustomCopilotKitMiddleware()],
        checkpointer=checkpointer,
        name="ask_user_frontend_test",
    )


def _ai_calling_ask_user(*, msg_id: str, questions: list[dict]) -> AIMessage:
    return AIMessage(
        id=msg_id,
        content="",
        tool_calls=[
            {"name": "ask_user_questions", "args": {"questions": questions}, "id": "call_ask"}
        ],
    )


# ---------------------------------------------------------------------------
# 1. Single-question pause
# ---------------------------------------------------------------------------


async def test_single_question_pauses_with_frontend_tool_calls_payload():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "ask-pause-1"}}
    model = MockChatModel(responses=[
        _ai_calling_ask_user(
            msg_id="ai1",
            questions=[{"question": "Which book did you mean?", "options": []}],
        ),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add the book")],
            "copilotkit": {"actions": [ASK_USER_DESCRIPTOR]},
        },
        config=config,
    )

    state = graph.get_state(config)
    assert state.tasks
    interrupts = state.tasks[0].interrupts
    assert len(interrupts) == 1

    value = interrupts[0].value
    calls = value.get("frontend_tool_calls")
    assert isinstance(calls, list) and len(calls) == 1
    call = calls[0]
    assert call["name"] == "ask_user_questions"
    assert call["id"] == "call_ask"
    assert call["args"]["questions"][0]["question"] == "Which book did you mean?"


# ---------------------------------------------------------------------------
# 2. Single-question resume — answers come back as the tool result
# ---------------------------------------------------------------------------


async def test_single_question_resume_threads_answers_to_tool_result():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "ask-resume-1"}}
    model = MockChatModel(responses=[
        _ai_calling_ask_user(
            msg_id="ai1",
            questions=[{"question": "Which book?", "options": []}],
        ),
        AIMessage(id="ai2", content="Added 'The Pragmatic Programmer'."),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add the book")],
            "copilotkit": {"actions": [ASK_USER_DESCRIPTOR]},
        },
        config=config,
    )
    await graph.ainvoke(
        Command(resume={"tool_results": [
            {"toolCallId": "call_ask", "content": '["The Pragmatic Programmer"]'},
        ]}),
        config=config,
    )

    state = graph.get_state(config)
    assert not state.tasks or all(not t.interrupts for t in state.tasks)

    messages = state.values["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "The Pragmatic Programmer" in tool_messages[0].content

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    final = next((m for m in reversed(ai_messages) if m.content), None)
    assert final is not None
    assert final.content == "Added 'The Pragmatic Programmer'."


# ---------------------------------------------------------------------------
# 3. Multi-question batch — all questions on one interrupt
# ---------------------------------------------------------------------------


async def test_multi_question_batch_shows_all_questions_on_one_interrupt():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "ask-multi"}}
    model = MockChatModel(responses=[
        _ai_calling_ask_user(
            msg_id="ai1",
            questions=[
                {"question": "Use existing or create new?", "options": ["Use existing", "Create new"]},
                {"question": "Which list?", "options": ["TBR", "Finished"]},
                {"question": "Notes?", "options": []},
            ],
        ),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add this book")],
            "copilotkit": {"actions": [ASK_USER_DESCRIPTOR]},
        },
        config=config,
    )

    state = graph.get_state(config)
    interrupts = state.tasks[0].interrupts
    assert len(interrupts) == 1
    calls = interrupts[0].value["frontend_tool_calls"]
    assert len(calls) == 1, "ask_user_questions is one tool call carrying many questions"
    questions = calls[0]["args"]["questions"]
    assert [q["question"] for q in questions] == [
        "Use existing or create new?",
        "Which list?",
        "Notes?",
    ]
    assert questions[0]["options"] == ["Use existing", "Create new"]


# ---------------------------------------------------------------------------
# 4. Mixed with another frontend tool — both pause together, both resume.
# ---------------------------------------------------------------------------


async def test_ask_user_batched_with_other_frontend_tool_in_one_interrupt():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "ask-mixed"}}
    model = MockChatModel(responses=[
        AIMessage(
            id="ai1",
            content="",
            tool_calls=[
                {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_add"},
                {"name": "ask_user_questions", "args": {"questions": [
                    {"question": "Anything else?", "options": []},
                ]}, "id": "call_ask"},
            ],
        ),
        AIMessage(id="ai2", content="Done."),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add an apple then check")],
            "copilotkit": {"actions": [ADD_ITEM_DESCRIPTOR, ASK_USER_DESCRIPTOR]},
        },
        config=config,
    )

    interrupts = graph.get_state(config).tasks[0].interrupts
    calls = interrupts[0].value["frontend_tool_calls"]
    assert {c["name"] for c in calls} == {"addItem", "ask_user_questions"}

    await graph.ainvoke(
        Command(resume={"tool_results": [
            {"toolCallId": "call_add", "content": '{"ok":true}'},
            {"toolCallId": "call_ask", "content": '["No"]'},
        ]}),
        config=config,
    )

    state = graph.get_state(config)
    assert not state.tasks or all(not t.interrupts for t in state.tasks)
    by_id = {m.tool_call_id: m.content for m in state.values["messages"] if isinstance(m, ToolMessage)}
    assert "ok\":true" in by_id["call_add"]
    assert "No" in by_id["call_ask"]
