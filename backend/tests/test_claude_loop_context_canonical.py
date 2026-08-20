"""Client key ordering must not bust the prompt cache.

The iOS client builds its ambient-context payloads from Swift `Dictionary`s,
whose iteration order is randomised — so the same snapshot arrives with shuffled
keys every turn. That block lands in the **system** prompt, which precedes
`messages` in Anthropic's cache prefix, so each reshuffle re-cached the whole
transcript behind it (observed: ~11k of avoidable `cache_write` per turn).
"""

from __future__ import annotations

import uuid

from ag_ui.core.types import Context, RunAgentInput, UserMessage

from pupa_backend.harnesses.claude.endpoint import (
    _canonical_json,
    _context_pairs,
    _render_context,
)


def _ctx(*values: str) -> list[Context]:
    return [Context(description=f"entry {i}", value=v) for i, v in enumerate(values)]


# --------------------------------------------------------------------------- #
# _canonical_json
# --------------------------------------------------------------------------- #

def test_shuffled_object_keys_canonicalise_to_the_same_bytes() -> None:
    a = '{"typeId":"tracker","myAppName":"WebExplorer"}'
    b = '{"myAppName":"WebExplorer","typeId":"tracker"}'
    assert _canonical_json(a) == _canonical_json(b)


def test_nested_objects_inside_arrays_are_sorted_too() -> None:
    """The skills roster is an array of objects — the shuffle happens inside."""
    a = '{"skills":[{"description":"d","when_to_use":"w","name":"n"}]}'
    b = '{"skills":[{"when_to_use":"w","name":"n","description":"d"}]}'
    assert _canonical_json(a) == _canonical_json(b)
    assert _canonical_json(a) == '{"skills":[{"description":"d","name":"n","when_to_use":"w"}]}'


def test_array_order_is_preserved() -> None:
    """Only object keys are sorted — a reordered list is a real change."""
    assert _canonical_json('{"paths":["b.md","a.md"]}') == '{"paths":["b.md","a.md"]}'


def test_non_json_values_pass_through_untouched() -> None:
    prose = "Live canvas state — thin enum, not JSON at all."
    assert _canonical_json(prose) == prose
    assert _canonical_json("") == ""


def test_json_scalars_pass_through_untouched() -> None:
    """A bare `5` or `true` parses as JSON but isn't a payload to normalise."""
    for scalar in ("5", "true", "null", '"just a string"'):
        assert _canonical_json(scalar) == scalar


def test_unicode_is_not_escaped_away() -> None:
    assert _canonical_json('{"name":"café — naïve"}') == '{"name":"café — naïve"}'


# --------------------------------------------------------------------------- #
# …applied to the rendered prompt
# --------------------------------------------------------------------------- #

def test_rendered_context_is_identical_across_a_key_reshuffle() -> None:
    first = _render_context(_ctx('{"typeId":"tracker","myAppName":"WebExplorer"}'))
    second = _render_context(_ctx('{"myAppName":"WebExplorer","typeId":"tracker"}'))
    assert first == second


def test_context_pairs_canonicalise_values_but_not_descriptions() -> None:
    pairs = _context_pairs(_ctx('{"b":1,"a":2}'))
    assert pairs == [("entry 0", '{"a":2,"b":1}')]


def test_a_real_content_change_still_shows_up() -> None:
    """Canonicalising must not paper over an actual edit."""
    first = _render_context(_ctx('{"components":[]}'))
    second = _render_context(_ctx('{"components":[{"id":"a"}]}'))
    assert first != second


def test_options_build_fingerprint_is_stable_across_a_reshuffle() -> None:
    """End to end: the same snapshot with shuffled keys is one cache prefix."""
    from pupa_backend.harnesses.claude import usage

    def fp(value: str):
        pairs = _context_pairs(_ctx(value))
        return usage.fingerprint(
            model="haiku", base_system="base",
            system=f"base\n\n{_render_context(_ctx(value))}",
            context_pairs=pairs, tool_specs=[], permission_mode="default",
            thinking={}, skills=None, cwd=None,
        )

    assert fp('{"typeId":"tracker","myAppName":"WebExplorer"}') == fp(
        '{"myAppName":"WebExplorer","typeId":"tracker"}'
    )


def test_run_input_context_flows_through_canonicalised() -> None:
    """`RunAgentInput.context` is the real shape the endpoint reads."""
    inp = RunAgentInput(
        thread_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()), state={},
        messages=[UserMessage(id="u1", content="hi")], tools=[], forwarded_props={},
        context=_ctx('{"z":1,"a":{"y":2,"b":3}}'),
    )
    assert _context_pairs(inp.context) == [("entry 0", '{"a":{"b":3,"y":2},"z":1}')]
