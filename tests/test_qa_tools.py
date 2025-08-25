from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path for direct imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from twin_generator.tools.qa_tools import _check_asset  # noqa: E402


def test_check_asset_true_when_no_assets() -> None:
    assert _check_asset() is True
    assert _check_asset("", "") is True


def test_check_asset_false_when_graph_missing(tmp_path: Path) -> None:
    # Provide a path that does not exist while table_html is missing
    missing = tmp_path / "nonexistent.png"
    assert _check_asset(str(missing), None) is False
