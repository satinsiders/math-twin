import inspect
import sys
import typing
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.tool import _INTROSPECTION_CACHE, function_tool  # noqa: E402


def test_annotation_to_type_handles_complex_signatures():
    def f(a: int, b: Optional[float], c: list[str], d: Union[int, float], e: Union[int, str]):
        pass

    tool = function_tool(f)
    props = tool["parameters"]["properties"]
    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "array"
    assert props["d"]["type"] == "number"
    assert props["e"]["type"] == "string"


def test_function_tool_caches_introspection():
    def g(x: int) -> int:
        return x

    _INTROSPECTION_CACHE.clear()
    with patch("agents.tool.inspect.signature", wraps=inspect.signature) as sig, \
         patch("agents.tool.get_type_hints", wraps=typing.get_type_hints) as hints:
        function_tool(g)
        function_tool(g)
        assert sig.call_count == 1
        assert hints.call_count == 1
