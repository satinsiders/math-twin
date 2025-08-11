"""Runner interface for executing agents synchronously."""

from typing import Any


class Runner:
    @staticmethod
    def run_sync(agent: Any, input: Any) -> Any:
        """Execute the given agent with the provided input."""
        raise NotImplementedError
