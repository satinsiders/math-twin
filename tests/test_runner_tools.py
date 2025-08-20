from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.run import Runner  # noqa: E402


def test_sanitize_tools_caches_by_name() -> None:
    Runner._SANITIZED_CACHE.clear()

    def func() -> str:
        return "ok"

    tool1 = {"name": "foo", "_func": func, "desc": "1"}
    tool2 = {"name": "foo", "_func": func, "desc": "2"}

    sanitized1, _ = Runner._sanitize_tools([tool1])
    sanitized2, _ = Runner._sanitize_tools([tool2])

    assert sanitized1[0] == sanitized2[0]
    assert sanitized1[0] is not sanitized2[0]


def test_execute_tool_calls_runs_tools_and_returns_response() -> None:
    def add(x: int, y: int) -> int:
        return x + y

    call = SimpleNamespace(
        id="1",
        function=SimpleNamespace(name="adder", arguments=json.dumps({"x": 1, "y": 2})),
    )
    resp = SimpleNamespace(
        status="requires_action",
        required_action=SimpleNamespace(
            submit_tool_outputs=SimpleNamespace(tool_calls=[call])
        ),
        id="resp1",
    )
    final_resp = SimpleNamespace(status="done", output_text="ok")

    class FakeResponses:
        def submit_tool_outputs(self, response_id: str, tool_outputs: list[dict[str, str]]) -> SimpleNamespace:
            final_resp.received = tool_outputs
            return final_resp

    client = SimpleNamespace(responses=FakeResponses())
    tool_map = {"adder": {"_func": add}}

    result = Runner._execute_tool_calls(client, resp, tool_map)
    assert result is final_resp
    assert final_resp.received == [
        {"tool_call_id": "1", "output": "3"}
    ]


def test_execute_tool_calls_aborts_when_status_never_resolves() -> None:
    resp = SimpleNamespace(
        status="requires_action",
        required_action=SimpleNamespace(
            submit_tool_outputs=SimpleNamespace(tool_calls=[])
        ),
        id="resp1",
    )

    class FakeResponses:
        def __init__(self) -> None:
            self.count = 0

        def submit_tool_outputs(self, response_id: str, tool_outputs: list[Any]) -> SimpleNamespace:
            self.count += 1
            return resp

    client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(RuntimeError) as excinfo:
        Runner._execute_tool_calls(client, resp, {})

    assert "requires_action" in str(excinfo.value)
    assert client.responses.count == 10


def test_execute_tool_calls_raises_on_invalid_json() -> None:
    def noop() -> None:
        pass

    call = SimpleNamespace(
        id="1",
        function=SimpleNamespace(name="noop", arguments="{"),
    )
    resp = SimpleNamespace(
        status="requires_action",
        required_action=SimpleNamespace(
            submit_tool_outputs=SimpleNamespace(tool_calls=[call])
        ),
        id="resp1",
    )

    client = SimpleNamespace(
        responses=SimpleNamespace(
            submit_tool_outputs=lambda *args, **kwargs: resp
        )
    )

    with pytest.raises(RuntimeError) as excinfo:
        Runner._execute_tool_calls(client, resp, {"noop": {"_func": noop}})

    assert "tool noop provided invalid JSON arguments" in str(excinfo.value)
