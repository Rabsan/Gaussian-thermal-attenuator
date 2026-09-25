#!/usr/bin/env python3
"""Compatibility launcher for the modular coherent-information evaluator."""

from coherent_information import (
    exact, ball, encode, decimal_lower_bound, physical_probability, h2,
    word, entropy, evaluate, load_state, main,
)


if __name__ == "__main__":
    raise SystemExit(main())
