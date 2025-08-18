from __future__ import annotations

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

