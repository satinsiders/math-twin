from __future__ import annotations

import pytest

from twin_generator.utils import safe_json


def test_safe_json_single_quotes() -> None:
    out = safe_json("{'a': 1}")
    assert out == {"a": 1}


def test_safe_json_trailing_comma() -> None:
    out = safe_json('{"a": 1,}')
    assert out == {"a": 1}


def test_safe_json_unbalanced_braces() -> None:
    out = safe_json('{"a": 1')
    assert out == {"a": 1}


def test_safe_json_combined_issues() -> None:
    out = safe_json("{'a': 1,")
    assert out == {"a": 1}


def test_safe_json_with_comments() -> None:
    out = safe_json('{"a":1,// c\n"b":2}')
    assert out == {"a": 1, "b": 2}


def test_safe_json_apostrophes_inside_strings() -> None:
    out = safe_json("{'text': \"it's great\"}")
    assert out == {"text": "it's great"}


def test_safe_json_error_message() -> None:
    with pytest.raises(ValueError) as exc:
        safe_json("not json")
    msg = str(exc.value)
    assert "Original snippet: not json" in msg

