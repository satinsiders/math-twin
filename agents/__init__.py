from __future__ import annotations

from typing import Optional


class Agent:
    def __init__(self, name: str, instructions: str, model: Optional[str] = None) -> None:
        self.name: str = name
        self.instructions: str = instructions
        self.model: Optional[str] = model
