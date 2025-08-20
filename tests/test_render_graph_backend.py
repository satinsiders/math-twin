import json
import importlib
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_render_graph_headless_uses_agg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)

    import twin_generator.tools as tools
    importlib.reload(tools)

    path = tools._render_graph(json.dumps({"points": [[0, 0], [1, 1]]}))
    try:
        assert Path(path).is_file()
        import matplotlib
        assert matplotlib.get_backend().lower() == "agg"
    finally:
        Path(path).unlink(missing_ok=True)


def test_missing_gui_backend_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "tkagg")
    monkeypatch.delenv("DISPLAY", raising=False)

    import matplotlib
    original_use = matplotlib.use

    def fail_use(backend: str, *args: Any, **kwargs: Any) -> Any:
        if backend == "TkAgg":
            raise ImportError("TkAgg not available")
        return original_use(backend, *args, **kwargs)

    original_use("pdf")
    monkeypatch.setattr(matplotlib, "use", fail_use)
    import twin_generator.tools as tools
    with pytest.warns(RuntimeWarning):
        importlib.reload(tools)

    assert matplotlib.get_backend().lower() == "agg"
    path = tools._render_graph(json.dumps({"points": []}))
    Path(path).unlink(missing_ok=True)
