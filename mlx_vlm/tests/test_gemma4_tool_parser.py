"""Tests for the Gemma 4 tool-call parser."""

import pytest

from mlx_vlm.tool_parsers.gemma4 import parse_tool_call

TC_START = "<|tool_call>"
TC_END = "<tool_call|>"
ESC = '<|"|>'


def _wrap(call_body: str) -> str:
    return f"{TC_START}{call_body}{TC_END}"


def _str(value: str) -> str:
    return f"{ESC}{value}{ESC}"


@pytest.mark.parametrize(
    ("call_body", "expected"),
    [
        (
            f"call:get_weather{{city:{_str('Nuremberg')}}}",
            {"name": "get_weather", "arguments": {"city": "Nuremberg"}},
        ),
        (
            "call:set_volume{level:42}",
            {"name": "set_volume", "arguments": {"level": 42}},
        ),
        (
            f"call:lobe-web-browsing____search{{search_query:{_str('Lustgarten Berlin')}}}",
            {
                "name": "lobe-web-browsing____search",
                "arguments": {"search_query": "Lustgarten Berlin"},
            },
        ),
        (
            f"call:get-weather{{city:{_str('Tokyo')}}}",
            {"name": "get-weather", "arguments": {"city": "Tokyo"}},
        ),
        (
            f"call:edit-file{{path:{_str('test.txt')},edits:[{{newText:{_str('orange')},oldText:{_str('apple')}}}]}}",
            {
                "name": "edit-file",
                "arguments": {
                    "path": "test.txt",
                    "edits": [{"newText": "orange", "oldText": "apple"}],
                },
            },
        ),
    ],
)
def test_parse_tool_call_accepts_expected_function_names(call_body, expected):
    assert parse_tool_call(_wrap(call_body)) == expected


@pytest.mark.parametrize(
    "call_body",
    [
        f"call:get-weather-e\u00e9{{city:{_str('Paris')}}}",
        f"call:get.weather{{city:{_str('Paris')}}}",
        f"call:{'a' * 65}{{city:{_str('Paris')}}}",
    ],
)
def test_parse_tool_call_rejects_invalid_function_names(call_body):
    with pytest.raises(ValueError, match="No function call found"):
        parse_tool_call(_wrap(call_body))


def test_parse_tool_call_raises_when_no_tool_call_is_present():
    with pytest.raises(ValueError, match="No function call found"):
        parse_tool_call("just a normal model response, no tool call here")
