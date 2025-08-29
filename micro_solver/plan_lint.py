from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .plan_policy import lint_plan


def _load_input(path: str) -> Any:
    data = sys.stdin.read() if path == "-" else open(path, "r", encoding="utf-8").read()
    try:
        obj = json.loads(data)
    except Exception as exc:
        raise SystemExit(f"invalid json: {exc}")
    # Accept either {"plan_steps": [...]} or just [...]
    if isinstance(obj, dict) and "plan_steps" in obj:
        return obj["plan_steps"]
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lint micro-solver plan steps for policy compliance")
    p.add_argument("path", help="JSON file path or '-' for stdin")
    args = p.parse_args(argv)

    steps = _load_input(args.path)
    res = lint_plan(steps)
    ok = bool(res.get("ok", False))
    if ok:
        print("ok")
        return 0
    issues = res.get("issues", [])
    if issues:
        for msg in issues:
            print(msg)
    else:
        print("plan-policy-violation")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

