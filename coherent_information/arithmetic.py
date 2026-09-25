"""Exact scalar conversion and interval-arithmetic helpers."""

from decimal import Decimal, localcontext, ROUND_FLOOR
from fractions import Fraction
import math

from ._flint import arb, ctx


def exact(value):
    """Exact rational interpretation: decimal strings, fractions, or dyadics."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        token = value.strip()
        if "0x" in token.lower():
            # Parse hexadecimal floats without first rounding to binary64.
            sign = -1 if token.startswith("-") else 1
            token = token.lstrip("+-").lower()
            mantissa, exponent = token[2:].split("p")
            whole, _, frac = mantissa.partition(".")
            numerator = int((whole or "0") + frac, 16)
            shift = int(exponent) - 4*len(frac)
            return sign * Fraction(numerator) * Fraction(2)**shift
        return Fraction(token)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite coefficient or parameter")
        return Fraction.from_float(value)
    return Fraction(value)


def ball(value):
    value = exact(value)
    return arb(value.numerator) / arb(value.denominator)


def encode(value):
    # Preserve enough digits to enclose all requested working precision.
    return value.str(max(60, int(ctx.prec*0.30103)+8), more=True)


def decimal_lower_bound(value):
    """A displayed decimal rounded DOWN, so printing cannot weaken rigor."""
    rational = value.lower().fmpq()
    with localcontext() as context:
        context.prec = 40
        context.rounding = ROUND_FLOOR
        return str(Decimal(int(rational.numerator))/Decimal(int(rational.denominator)))


def physical_probability(value):
    if not value.is_finite() or value.upper() < 0 or value.lower() > 1:
        raise ArithmeticError("probability enclosure inconsistent with [0,1]")
    return value.intersection(arb(0).union(arb(1)))


def h2(value):
    """Enclose binary entropy even when its input interval touches 0 or 1."""
    value = physical_probability(value)
    lo, hi = max(arb(0), value.lower()), min(arb(1), value.upper())
    def at(x):
        if x == 0 or x == 1:
            return arb(0)
        return -(x*x.log()+(1-x)*(1-x).log())/arb(2).log()
    left, right = at(lo), at(hi)
    lower = min(left.lower(), right.lower())
    upper = arb(1) if lo <= arb(1)/2 <= hi else max(left.upper(), right.upper())
    return lower.union(upper)
