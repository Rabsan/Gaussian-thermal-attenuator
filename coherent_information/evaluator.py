"""Gaussian thermal-attenuator coherent-information evaluation."""

import hashlib
import json
import math
import platform
import sys
import time

from ._flint import flint, arb, arb_mat, ctx
from .arithmetic import (
    exact, ball, encode, decimal_lower_bound, physical_probability, h2,
)
from .entropy import entropy
from .states import word


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
