import json
from pathlib import Path

import pytest

import twin_generator.tools as tools


def test_render_graph_raises_for_insufficient_point_entries() -> None:
    spec = json.dumps({"points": [[1]]})
    with pytest.raises(ValueError, match=r"expected \[x, y\]"):
        tools._render_graph(spec)


def test_render_graph_truncates_extra_point_entries() -> None:
    spec = json.dumps({"points": [[0, 1, 2], [2, 3, 4]]})
    path = tools._render_graph(spec)
    try:
        assert Path(path).is_file()
    finally:
        Path(path).unlink(missing_ok=True)
