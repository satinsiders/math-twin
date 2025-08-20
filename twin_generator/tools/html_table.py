"""HTML table rendering helpers."""
from __future__ import annotations

import html as _html
import json

from agents.tool import function_tool

__all__ = ["make_html_table_tool", "_make_html_table"]


def _make_html_table(table_json: str) -> str:
    """Convert a JSON table spec → `<table>` element string (values escaped)."""
    data = json.loads(table_json)
    header = data.get("header", [])
    rows = data.get("rows", [])

    head_html = "".join(f"<th>{_html.escape(str(h))}</th>" for h in header)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>"


make_html_table_tool = function_tool(_make_html_table)
