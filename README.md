# Gaussian thermal attenuator

This repository has the code and input states for the numerical result in

R. G. Ahmed, S. Bhalerao, S. Lee, F. Leditzky, D. Leung, L. Schaeffer, and G. Smith, "A depolarizing choir sings in Gaussian harmony", [arXiv:2609.39747](https://arxiv.org/abs/2609.39747) (2026).

The code computes the coherent information of a rank-two input state $\rho = q|\psi_0\rangle\langle\psi_0| + (1-q)|\psi_1\rangle\langle\psi_1|$ through the Gaussian channel $\mathcal{A}_G \circ \mathcal{L}_T$, with $T = 2\eta^2/(1+\eta)$ and $G = (1+\eta)/(2\eta)$. The codeword $\psi_0$ is supported on Fock states $n \equiv 0 \pmod 3$ and $\psi_1$ on $n \equiv 1 \pmod 3$. All arithmetic uses Arb intervals and verified eigenvalues. A run that reports `positive_certified` proves that the coherent information of the full, untruncated channel is positive.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install python-flint==0.9.0
```

## Reproducing the paper's result

The paper uses the state in `eta7294452_state.csv` at $\eta = 0.7294452$. This corresponds to depolarizing noise $p = 3(1-\eta)/4 = 0.2029161$. Run this from the repository folder:

```sh
python -m coherent_information \
  --state eta7294452_state.csv \
  --eta 0.7294452 --prior 0.4506151671134594 \
  --output results/eta7294452_report.json
```

The command prints a summary and writes the full report, including every eigenvalue enclosure, to `results/eta7294452_report.json`.

- The code does not read the `mixture_p` column of the CSV, so pass $q$ with `--prior`.
- The `eta` column in the CSV is 0.729445199800, which is $p = 0.2029161002$. You can pass that value to `--eta` instead.
- The defaults are amplifier cutoff `Q=100` and 1024 bits of precision. This state has 221 Fock levels, so the run takes longer than the earlier example below.
- `python -m coherent_information --help` lists the other options.

## Earlier example

`2026-09-16-1307-DL-mod3_eta_0p72990_codewords.csv` is an earlier state at $\eta = 0.7299$, which is $p = 0.202575$. The paper does not use it. Its certified report is `results/report.json`.

```sh
python -m coherent_information \
  --state 2026-09-16-1307-DL-mod3_eta_0p72990_codewords.csv \
  --eta 0.7299 --prior 0.45683956664883835 \
  --output results/report.json
```

## Files

- `eta7294452_state.csv`: input codewords for the paper's result.
- `2026-09-16-1307-DL-mod3_eta_0p72990_codewords.csv`: input codewords for the earlier example.
- `coherent_information/evaluator.py`: channel calculation and certified coherent-information bounds.
- `coherent_information/entropy.py`: verified eigenvalues and entropy bounds.
- `coherent_information/arithmetic.py`: exact scalar conversion and interval-arithmetic helpers.
- `coherent_information/states.py`: codeword validation and JSON, CSV, or NPZ loading.
- `coherent_information/cli.py`, `__main__.py`: command-line options and module entry point.
- `coherent_information/__init__.py`, `_flint.py`: Python API exports and FLINT imports.
- `coherent_information_1.py`: alternative script launcher.
- `results/`: saved certification reports.
