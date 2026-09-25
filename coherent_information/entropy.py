"""Rigorous spectral certification and entropy enclosures."""

from ._flint import flint, arb, arb_mat
from .arithmetic import encode


def entropy(matrix, label):
    dim = matrix.nrows()
    if dim == 0:
        return arb(0), {"dimension": 0, "entropy_bits": "0", "eigenvalues": []}
    # Remove only rows/columns proved identically zero, and explicitly record
    # their exact zero eigenvalues. This handles pure-input endpoint priors
    # without asking the eigensolver to isolate a large structural nullspace.
    active = [i for i in range(dim) if any(matrix[i,j] != 0 for j in range(dim))]
    if len(active) == dim:
        reduced = matrix
    else:
        reduced = arb_mat([[matrix[i,j] for j in active] for i in active]) if active else None
    eigenvalues = (reduced.eig(algorithm="rump", multiple=True, nonstop=False)
                   if active else [])
    eigenvalues += [flint.acb(0)]*(dim-len(active))
    if len(eigenvalues) != dim:
        raise ArithmeticError(f"{label}: incomplete spectral multiplicities")
    total, entries, touches_zero = arb(0), [], 0
    ln2 = arb(2).log()
    for ev in eigenvalues:
        if not (ev.real.is_finite() and ev.imag.is_finite() and ev.imag.contains(0)):
            raise ArithmeticError(f"{label}: invalid rigorous eigenvalue enclosure")
        x = ev.real
        if x.upper() < 0 or x.lower() > 1:
            raise ArithmeticError(f"{label}: spectral enclosure contradicts Gram positivity")
        lo, hi = max(arb(0), x.lower()), min(arb(1), x.upper())
        if lo == 0:
            touches_zero += 1
            if hi == 0:
                term = arb(0)
            elif hi < (-arb(1)).exp():
                term = arb(0).union(-hi*hi.log()/ln2)
            else:
                term = arb(0).union(1/(arb(1).exp()*ln2))
        else:
            y = lo.union(hi)
            term = -y*y.log()/ln2
        total += term
        entries.append({"real": encode(x), "imaginary": encode(ev.imag), "entropy": encode(term)})
    if not total.is_finite():
        raise ArithmeticError(f"{label}: nonfinite entropy enclosure")
    return total, {"dimension": dim, "entropy_bits": encode(total),
                   "zero_crossing_enclosures": touches_zero, "eigenvalues": entries}
