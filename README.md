# Gaussian thermal attenuator

Python code to compute coherent information for finite, real mod-3 codes through a Gaussian thermal attenuator.
Uses Arb interval arithmetic and verified eigenvalues to certify a lower bound for the full channel.

## Quick start

From this folder, install the dependency and evaluate the included CSV:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install python-flint==0.9.0

python -m coherent_information \
  --state 2026-09-16-1307-DL-mod3_eta_0p72990_codewords.csv \
  --eta 0.7299 --prior 0.45683956664883835 \
  --output results/report.json
```

This uses the default cutoff `Q=100` and precision of 1024 bits. It prints a summary and saves the full certification report to `results/report.json`. Use `python -m coherent_information --help` for other options.

## Files

- `2026-09-16-1307-DL-mod3_eta_0p72990_codewords.csv`: Input codewords for the example above.
- `coherent_information/evaluator.py`: Channel calculation and certified coherent-information bounds.
- `coherent_information/entropy.py`: Verified eigenvalues and entropy bounds.
- `coherent_information/arithmetic.py`: Exact scalar conversion and interval-arithmetic helpers.
- `coherent_information/states.py`: Codeword validation and JSON, CSV, or NPZ loading.
- `coherent_information/cli.py`, `__main__.py`: Command-line options and module entry point.
- `coherent_information/__init__.py`, `_flint.py`: Python API exports and FLINT imports.
- `coherent_information_1.py`: Alternative script launcher.
- `results/`: Saved certification reports, including the supplied CSV's positive result.
