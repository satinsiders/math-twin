"""Public package interface for the Twin Math‑Problem Generator.

Importing this package gives you easy access to the top‑level helpers without
having to know the internal module layout.

Typical usage
-------------
>>> from twin_generator import generate_twin
>>> generate_twin(src_problem, src_solution)
"""
from importlib.metadata import version as _version  # type: ignore

from .pipeline import generate_twin  # re‑export for convenience

__all__ = [
    "generate_twin",
    "__version__",
]

try:
    __version__ = _version("twin_generator")
except Exception:  # pragma: no cover – package not installed yet
    __version__ = "0.0.0"
