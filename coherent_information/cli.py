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
import hashlib
import json
from pathlib import Path
import sys

from ._flint import arb, ctx
from .arithmetic import ball
from .evaluator import evaluate
from .states import load_state


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
