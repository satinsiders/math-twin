"""QA-specific tool wrappers."""
from __future__ import annotations

import json
import os
from typing import Any

from agents.tool import function_tool

from .calc import _sanitize_params
from ..utils import coerce_answers, validate_output

__all__ = [
    "sanitize_params_tool",
    "validate_output_tool",
    "check_asset_tool",
    "_sanitize_params_tool",
    "_validate_output_tool",
    "_check_asset",
]


def _sanitize_params_tool(params_json: str) -> dict[str, Any]:
    """Return numeric parameters and skipped keys from *params_json*."""
    params = json.loads(params_json)
    sanitized, skipped = _sanitize_params(params)
    sanitized_out = {k: str(v) for k, v in sanitized.items()}
    return {"sanitized": sanitized_out, "skipped": skipped}


sanitize_params_tool = function_tool(_sanitize_params_tool)
sanitize_params_tool["name"] = "sanitize_params_tool"


def _validate_output_tool(block_json: str) -> dict[str, Any]:
    """Coerce answer fields then validate the formatter output."""
    block = json.loads(block_json)
    block = coerce_answers(block)
    return validate_output(block)


validate_output_tool = function_tool(_validate_output_tool)
validate_output_tool["name"] = "validate_output_tool"


def _check_asset(graph_path: str | None = None, table_html: str | None = None) -> bool:
    """Return ``True`` if ``graph_path`` exists or ``table_html`` is present."""
    if graph_path and os.path.isfile(graph_path):
        return True
    if table_html and str(table_html).strip():
        return True
    return False


check_asset_tool = function_tool(_check_asset)
check_asset_tool["name"] = "check_asset_tool"
