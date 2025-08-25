import importlib.util
from pathlib import Path

# Load utils module directly to avoid triggering package imports
_utils_path = Path(__file__).resolve().parents[1] / "twin_generator" / "utils.py"
_spec = importlib.util.spec_from_file_location("tg_utils", _utils_path)
utils = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(utils)  # type: ignore[assignment]
_normalize_graph_points = utils._normalize_graph_points


def test_normalize_graph_points_converts_dicts() -> None:
    spec = {"points": [{"X": 0, "Y": 1}, {"x": "2", "y": 3}, [4, 5]]}
    _normalize_graph_points(spec)
    assert spec["points"] == [[0.0, 1.0], [2.0, 3.0], [4, 5]]


def test_normalize_graph_points_non_list_unchanged() -> None:
    spec = {"points": "oops"}
    _normalize_graph_points(spec)
    assert spec["points"] == "oops"
