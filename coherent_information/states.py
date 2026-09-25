"""Codeword validation and JSON, CSV, and NPZ state loading."""

import csv
from fractions import Fraction
import json
from pathlib import Path

from .arithmetic import exact


def word(values, residue):
    items = values.items() if hasattr(values, "items") else enumerate(values)
    result, last = {}, -1
    for n, value in items:
        index = int(n)
        if exact(n) != index or index < 0:
            raise ValueError("Fock indices must be nonnegative integers")
        if index in result:
            raise ValueError("duplicate Fock index")
        value = exact(value)
        if value and index % 3 != residue:
            raise ValueError(f"psi{residue} has a nonzero coefficient on the wrong residue")
        result[index] = value
        last = max(last, index)
    if not any(result.values()):
        raise ValueError("each codeword must have positive norm")
    return result, last


def load_state(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(), parse_float=str)
        return obj["psi0"], obj["psi1"], obj.get("prior")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("empty CSV")
        indices = [int(row["n"]) for row in rows]
        if len(set(indices)) != len(indices):
            raise ValueError("CSV has duplicate Fock indices")
        words = []
        for r in (0,1):
            key = f"psi{r}_raw_binary64_hex"
            if key not in rows[0]:
                key = f"psi{r}_amplitude"
            words.append({int(row["n"]): row[key] for row in rows})
        return *words, None
    if path.suffix.lower() == ".npz":
        try:
            import numpy as np
        except ImportError as exc:
            raise ValueError("Reading NPZ requires NumPy: python -m pip install numpy") from exc
        with np.load(path, allow_pickle=False) as obj:
            words = []
            for r in (0,1):
                support, values = obj[f"s{r}"], obj[f"a{r}"]
                if len(support) != len(values) or len(set(map(int,support))) != len(support):
                    raise ValueError("invalid NPZ support")
                if np.iscomplexobj(values) and np.any(values.imag != 0):
                    raise ValueError("only real amplitudes are supported")
                wordmap = {}
                for n, value in zip(support,values):
                    if int(n) != n or n < 0:
                        raise ValueError("invalid NPZ Fock index")
                    real_value = np.real(value)
                    # Preserve the exact stored floating value, including
                    # extended-precision NumPy floats, rather than casting.
                    if hasattr(real_value, "as_integer_ratio"):
                        numerator, denominator = real_value.as_integer_ratio()
                        wordmap[int(n)] = Fraction(int(numerator),int(denominator))
                    elif np.issubdtype(values.dtype, np.integer):
                        wordmap[int(n)] = Fraction(int(real_value))
                    else:
                        raise ValueError("NPZ amplitudes must be real integer or floating scalars")
                if "K" in obj.files:
                    K = int(obj["K"])
                    if K < max(wordmap) or K != obj["K"]:
                        raise ValueError("invalid NPZ K")
                    wordmap.setdefault(K,"0")
                words.append(wordmap)
            prior = obj["prior"] if "prior" in obj.files else obj["p"] if "p" in obj.files else None
            if prior is not None:
                value = prior.item()
                if hasattr(value, "as_integer_ratio"):
                    numerator, denominator = value.as_integer_ratio()
                    prior = Fraction(int(numerator),int(denominator))
                else:
                    prior = exact(value)
            return *words, prior
    raise ValueError("state file must be JSON, CSV, or NPZ")
