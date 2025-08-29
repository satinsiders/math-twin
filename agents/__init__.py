from __future__ import annotations

from typing import Optional

# Ensure submodules are importable via attribute access for test patches
# e.g., patch("agents.tool.inspect.signature").
from . import tool  # noqa: F401
from . import run  # noqa: F401


class Agent:
    def __init__(self, name: str, instructions: str, model: Optional[str] = None) -> None:
        self.name: str = name
        self.instructions: str = instructions
        self.model: Optional[str] = model
