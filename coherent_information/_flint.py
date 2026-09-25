"""Shared FLINT imports and the original dependency check."""

try:
    import flint
    from flint import arb, arb_mat, ctx
except ImportError as exc:
    raise SystemExit("Install the one required package: python -m pip install python-flint==0.9.0") from exc
