import pathlib
import sys
import json
from types import SimpleNamespace

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import micro_solver.agents as A
from micro_solver.state import MicroState
from micro_solver.steps_recognition import _micro_normalize, _micro_tokenize
from micro_solver import orchestrator as orch
from micro_solver.orchestrator import MicroRunner, MicroGraph


def test_tokenize_includes_tokens_per_sentence(monkeypatch):
    def fake_invoke(agent, payload, qa_feedback=None):
        if agent is A.TokenizerAgent:
            return ({"sentences": ["hello world", "bye"], "tokens": ["hello", "world", "bye"]}, None)
        return ({}, "err")

    monkeypatch.setattr("micro_solver.steps_recognition._invoke", fake_invoke)

    captured = {}

    def fake_run_sync(agent, input):
        captured["payload"] = json.loads(input)
        return SimpleNamespace(final_output="pass")

    monkeypatch.setattr(orch.AgentsRunner, "run_sync", fake_run_sync)

    runner = MicroRunner(MicroGraph([_micro_normalize, _micro_tokenize]))
    state = MicroState(problem_text="hello world. bye")
    runner.run(state)

    out = captured["payload"]["out"]
    assert out["tokens_per_sentence"] == [["hello", "world"], ["bye"]]
