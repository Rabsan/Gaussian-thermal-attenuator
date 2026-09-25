"""Rigorous coherent information for real mod-3 codes."""

from .arithmetic import (
    exact, ball, encode, decimal_lower_bound, physical_probability, h2,
)
from .states import word, load_state
from .entropy import entropy
from .evaluator import evaluate
from .cli import main

__all__ = [
    "exact", "ball", "encode", "decimal_lower_bound", "physical_probability",
    "h2", "word", "load_state", "entropy", "evaluate", "main",
]
