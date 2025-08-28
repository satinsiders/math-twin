from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path for direct imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from twin_generator.tools.qa_tools import _check_asset  # noqa: E402


def test_qa_tools_validate_and_run(monkeypatch):
    import types
    from twin_generator.pipeline_helpers import AgentsRunner, _QA_TOOLS  # noqa: E402
    from twin_generator.agents import QAAgent  # noqa: E402

    sanitized: list[dict[str, object]] = []

    class DummyClient:
        def __init__(self) -> None:
            self.responses = self

        def create(self, **kwargs):
            sanitized.extend(kwargs.get("tools", []))
            return types.SimpleNamespace(status="completed", output_text="pass")

    fake_openai = types.SimpleNamespace(OpenAI=lambda: DummyClient())
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    res = AgentsRunner.run_sync(QAAgent, input="{}", tools=_QA_TOOLS)

    assert res.final_output == "pass"
    assert sanitized and all("type" in t for t in sanitized)


def test_check_asset_true_when_no_assets() -> None:
    assert _check_asset() is True
    assert _check_asset("", "") is True


def test_check_asset_false_when_graph_missing(tmp_path: Path) -> None:
    # Provide a path that does not exist while table_html is missing
    missing = tmp_path / "nonexistent.png"
    assert _check_asset(str(missing), None) is False
