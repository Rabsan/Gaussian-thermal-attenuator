#!/usr/bin/env python3
"""One-file, rigorous coherent-information evaluator for real mod-3 codes.

DEPENDENCY (no other project files, compiler, or executables are needed):
    python -m pip install python-flint==0.9.0

SUPPLY YOUR OWN STATE:
    python coherent_information.py --state state.json --eta 0.73 --Q 100
    python coherent_information.py --state state.csv --eta 0.732 --prior 0.45
    python coherent_information.py --state state.npz --eta 0.732

JSON format (use strings to specify exact decimals):
    {"psi0": {"0": "0.8", "3": "0.6"},
     "psi1": {"1": "0.6", "4": "0.8"}, "prior": "0.37"}
Alternatively psi0/psi1 can be full Fock arrays, with explicit zero entries.
CSV columns: n,psi0_amplitude,psi1_amplitude. The previously exported
psi0_raw_binary64_hex/psi1_raw_binary64_hex columns take precedence when
present. CSV requires --prior, since the mixing probability is not in it.
NPZ: a0,a1,s0,s1,K,prior (NumPy is needed ONLY to read this format).

PYTHON API:
    from coherent_information import evaluate
    result = evaluate({0: "0.8", 3: "0.6"}, {1: "0.6", 4: "0.8"},
                      eta="0.73", prior="0.37", Q=60, prec=1024)
Both codewords are normalized internally. psi0 occupies n=0 mod3 and psi1
occupies n=1 mod3; coefficients must be real. Arbitrary density matrices and
complex codewords are outside this evaluator's input format. prior is the
probability of psi0 in rho = prior|psi0><psi0|+(1-prior)|psi1><psi1|.

CHANNEL AND CERTIFICATION:
    N_eta = A_G after L_tau; tau=2 eta^2/(1+eta), G=(1+eta)/(2 eta).
    0 < eta <= 1. Q is an inclusive amplifier index; D keeps b=0,...,D-1.
The finite input has no omitted input tail. All scalar conversion, Gram
products, eigenvalue enclosures, and entropies use Arb interval arithmetic.
Rump verified eigenvalues with certified multiplicities are mandatory;
there is no approximate eigensolver or numerical eigenvalue cutoff.

Ic_retained_bits encloses S(B)-S(RB) for the NORMALIZED RETAINED output.
The infinite-output channel obeys the rigorous lower bound
    Ic_full >= t*Ic_retained - (1-t) - h2(delta_q),
where t is the retained trace and delta_q is the exact amplifier tail.
Full-channel positivity is reported only when this lower expression > 0.
--claim X additionally requires a strict proof Ic_full > X, or exits nonzero.
Increase --prec if eigenvalue certification fails or intervals are too wide;
increase Q/D if channel-tail corrections prevent resolving the sign.

Why the tail bound holds: flag the kept/omitted amplifier branches. Removing
the flag changes conditional entropy by at most h2(delta_q). Pinching the
receiver into kept/omitted Fock sectors cannot increase coherent information.
Every omitted branch has Ic >= -log2(dim R) = -1. Combining these facts gives
the stated bound. NB tail probabilities are evaluated by a finite binomial
identity, not by truncating a second infinite sum. Exact Gram positivity
allows intersecting spectral enclosures with [0,1]. An enclosure meeting
zero is bounded over its ENTIRE range; no eigenvalue is discarded.

Optional --output writes all spectral enclosures and certification evidence
to JSON. The ordinary summary prints to stdout; progress prints to stderr.
No example state, historical coefficient table, or prior result is embedded.

CLI defaults: Q=100,D=K+Q+1,prec=1024; --D can reduce the output space,
with its discarded weight included rigorously in the lower bound.
No global optimality claim is made.
"""

import argparse
import csv
from decimal import Decimal, localcontext, ROUND_FLOOR
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

try:
    import flint
    from flint import arb, arb_mat, ctx
except ImportError as exc:
    raise SystemExit("Install the one required package: python -m pip install python-flint==0.9.0") from exc


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


def evaluate(psi0, psi1, eta, prior="0.5", Q=60, D=None, prec=1024,
             threads=1, verbose=False):
    """Return rigorous interval strings and full-channel lower-bound evidence.

    psi0/psi1 are full Fock arrays or {Fock index: real amplitude} mappings.
    Decimal/fraction strings are recommended for exact intended parameters.
    The return value is JSON serializable and includes all eigenvalue bounds.
    FLINT uses a process-global context: do not call concurrently in threads.
    """
    eta, prior = exact(eta), exact(prior)
    if not 0 < eta <= 1:
        raise ValueError("eta must satisfy 0 < eta <= 1")
    if not 0 <= prior <= 1:
        raise ValueError("prior must satisfy 0 <= prior <= 1")
    if int(Q) != Q or Q < 0 or int(prec) != prec or prec < 64:
        raise ValueError("Q must be a nonnegative integer; prec must be >=64 bits")
    if int(threads) != threads or threads < 1:
        raise ValueError("threads must be a positive integer")
    Q = int(Q)
    raw0, K0 = word(psi0, 0)
    raw1, K1 = word(psi1, 1)
    K = max(K0, K1)
    D = K+Q+1 if D is None else D
    if int(D) != D or not 1 <= D <= K+Q+1:
        raise ValueError("D must be an integer in [1,K+Q+1]")
    D = int(D)
    canonical = {"eta": str(eta), "prior": str(prior), "K": K,
                 "psi0": {str(n): str(v) for n,v in sorted(raw0.items()) if v},
                 "psi1": {str(n): str(v) for n,v in sorted(raw1.items()) if v}}
    started = time.monotonic()
    def progress(*args):
        if verbose:
            print(f"[{time.monotonic()-started:.1f}s]", *args, file=sys.stderr, flush=True)
    old_threads = ctx.threads
    try:
        ctx.threads = int(threads)
        with ctx.workprec(int(prec)):
            a, support, norms = [], [], []
            for raw in (raw0, raw1):
                vector = [ball(raw.get(n, 0)) for n in range(K+1)]
                norm2 = sum((x*x for x in vector), arb(0))
                if not norm2 > 0:
                    raise ArithmeticError("normalization not resolved at this precision")
                a.append([x/norm2.sqrt() for x in vector])
                support.append([n for n in range(K+1) if raw.get(n, 0)])
                norms.append(norm2)
            e, p = ball(eta), ball(prior)
            tau, G = 2*e*e/(1+e), (1+e)/(2*e)
            z = (G-1)/G
            weights = [p, 1-p]
            roots = [x.sqrt() for x in weights]
            progress("building exact channel amplitudes", "K", K, "Q", Q, "D", D)
            loss = [[(arb(math.comb(n,l))*(1-tau)**l*tau**(n-l)).sqrt()
                     for l in range(n+1)] for n in range(K+1)]
            gain = [[(arb(math.comb(u+q,q))*(G-1)**q/G**(u+q+1)).sqrt()
                     for q in range(Q+1)] for u in range(K+1)]
            outputs = [list(range(r,D,3)) for r in range(3)]
            positions = [{b:i for i,b in enumerate(row)} for row in outputs]
            rb = []
            for syndrome in range(3):
                branches = [(l,q) for l in range(K+1) for q in range(Q+1)
                            if (q-l)%3 == syndrome]
                top = len(outputs[syndrome])
                W = arb_mat(top+len(outputs[(syndrome+1)%3]), len(branches))
                progress("Gram block", syndrome, W.nrows(), "by", W.ncols())
                for col,(l,q) in enumerate(branches):
                    for r in (0,1):
                        offset = 0 if r == 0 else top
                        for n in support[r]:
                            b = n-l+q
                            if n >= l and b < D:
                                W[offset+positions[(syndrome+r)%3][b], col] = (
                                    roots[r]*a[r][n]*loss[n][l]*gain[n-l][q])
                rb.append(W*W.transpose())
            trace = sum((M[i,i] for M in rb for i in range(M.nrows())), arb(0))
            if not trace > 0:
                raise ArithmeticError("retained trace is not certified positive; increase D/Q/precision")
            trace = physical_probability(trace)
            # Tail and crop probabilities are computed as sums of nonnegative
            # terms, not by subtracting nearly equal traces.
            photons = [sum((weights[r]*a[r][n]**2 for r in (0,1)), arb(0))
                       for n in range(K+1)]
            after_loss = [sum((photons[n]*loss[n][n-u]**2 for n in range(u,K+1)), arb(0))
                          for u in range(K+1)]
            tails = [sum((arb(math.comb(Q+u+1,j))*(1-z)**j*z**(Q+u+1-j)
                          for j in range(u+1)), arb(0)) for u in range(K+1)]
            dq = physical_probability(sum((after_loss[u]*tails[u] for u in range(K+1)), arb(0)))
            crop = physical_probability(sum((after_loss[u]*sum((gain[u][q]**2
                        for q in range(max(0,D-u),Q+1)), arb(0)) for u in range(K+1)), arb(0)))
            if not trace.overlaps(1-dq-crop):
                raise ArithmeticError("independent diagonal probability and Gram trace disagree")
            rb = [M/trace for M in rb]
            bblocks = []
            for r in range(3):
                dim, previous = len(outputs[r]), (r-1)%3
                off = len(outputs[previous])
                B = arb_mat(dim, dim)
                for i in range(dim):
                    for j in range(dim):
                        B[i,j] = rb[r][i,j]+rb[previous][off+i,off+j]
                bblocks.append(B)
            spectra = {}
            SB, SRB = arb(0), arb(0)
            for prefix, matrices in (("B", bblocks), ("RB", rb)):
                for r, M in enumerate(matrices):
                    label = prefix+str(r)
                    progress("certifying spectrum", label, M.nrows())
                    H, record = entropy(M, label)
                    spectra[label] = record
                    if prefix == "B":
                        SB += H
                    else:
                        SRB += H
            Ic = SB-SRB
            lower = trace*Ic-(1-trace)-h2(dq)
            if not (Ic.is_finite() and lower.is_finite()):
                raise ArithmeticError("nonfinite coherent-information enclosure")
            result = {"status": "positive_certified" if lower > 0 else "bound_computed_sign_unresolved",
                      "eta_exact": str(eta), "prior_exact": str(prior),
                      "tau_exact": str(2*eta*eta/(1+eta)), "G_exact": str((1+eta)/(2*eta)),
                      "K": K, "Q": Q, "D": D, "precision_bits": prec,
                      "python_flint_version": flint.__version__, "python_version": platform.python_version(),
                      "input_sha256": hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest(),
                      "input_definition": canonical, "raw_norms_squared": [encode(x) for x in norms],
                      "S_B_bits": encode(SB), "S_RB_bits": encode(SRB),
                      "Ic_retained_bits": encode(Ic), "retained_trace": encode(trace),
                      "delta_q": encode(dq), "delta_output_crop": encode(crop),
                      "h2_delta_q_bits": encode(h2(dq)),
                      "full_channel_lower_bound_bits": encode(lower),
                      "full_channel_lower_endpoint_bits": decimal_lower_bound(lower),
                      "bound_formula": "Ic_full >= t*Ic_retained - (1-t) - h2(delta_q)",
                      "eigensolver": "rump; multiple=True; nonstop=False",
                      "entropy_blocks": spectra, "runtime_seconds": time.monotonic()-started}
            if D == K+Q+1:
                # With no output crop, conditional-entropy concavity also
                # gives Ic_full <= t*Ic_retained + (1-t).
                result["full_channel_upper_bound_bits"] = encode(trace*Ic+(1-trace))
            return result
    finally:
        ctx.threads = old_threads


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True, help="JSON, CSV, or NPZ input state")
    parser.add_argument("--eta", required=True, help="exact channel parameter, e.g. 0.732 or 183/250")
    parser.add_argument("--prior", help="exact decimal or fraction; overrides the input's mixing probability")
    parser.add_argument("--Q", type=int)
    parser.add_argument("--D", type=int)
    parser.add_argument("--prec", type=int, default=1024, help="Arb working precision in bits")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--claim", help="require a strict full-channel lower bound greater than this number")
    parser.add_argument("--output", type=Path, help="write complete evidence including all eigenvalue enclosures")
    parser.add_argument("--quiet", action="store_true", help="suppress progress; still print the result")
    args = parser.parse_args()
    report = None
    try:
        a0,a1,p = load_state(args.state)
        p = args.prior if args.prior is not None else p
        if p is None:
            parser.error("this state file has no mixing probability; supply --prior")
        Q = args.Q if args.Q is not None else 100
        report = evaluate(a0,a1,args.eta,p,Q,args.D,args.prec,args.threads,not args.quiet)
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.claim is not None:
            with ctx.workprec(args.prec):
                passed = arb(report["full_channel_lower_bound_bits"]) > ball(args.claim)
            report["requested_strict_claim"] = args.claim
            report["requested_claim_proved"] = bool(passed)
            if not passed:
                raise ArithmeticError("requested full-channel strict lower bound was NOT proved; increase precision or cutoffs")
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
        print("Ic_retained_bits =",report["Ic_retained_bits"])
        print("full_channel_Ic_bits >=",report["full_channel_lower_endpoint_bits"])
        print("amplifier_tail_probability =",report["delta_q"])
        print("output_crop_probability =",report["delta_output_crop"])
        print("status =",report["status"])
        if args.claim:
            print("proved: full_channel_Ic_bits >",args.claim)
    except (ValueError, ArithmeticError, RuntimeError, KeyError, TypeError, OSError) as exc:
        if args.output:
            report = report or {}
            report.update({"status":"failed","error":str(exc)})
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
        print("CERTIFICATION FAILED:",str(exc),file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
