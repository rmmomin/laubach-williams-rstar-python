# LW Python Port

This repository exists to build a Python port of the Laubach-Williams (2003) natural-rate model using Bayesian state-space estimation. The core implementation now lives in `src/laubach_williams_rstar/model.py`, which combines custom Kalman filtering with PyMC priors that mirror the original literature.

## Environment setup

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

> Dependencies are pinned to versions that install cleanly on Python 3.9. If you upgrade to Python 3.10+ you can relax the caps to track the latest PyMC/PyTensor/pandas releases.

### Run the synthetic demo

`python scripts/run_example.py`

### Use your own data

Import `run_bayesian_lw` or `main` from `laubach_williams_rstar.model`, feed in your quarterly `log_gdp`, `pi`, and `r_real` arrays, and swap out the synthetic block in `model.py` (or supply your own driver script in `scripts/`).

## Project layout

```
├── requirements.txt
├── src/
│   └── laubach_williams_rstar/
│       ├── __init__.py
│       └── model.py
├── outputs/
│   ├── data/
│   └── figures/
└── scripts/
    └── run_example.py

Running `python scripts/run_example.py` now drops a CSV of smoothed states under `outputs/data/` and a PNG of the three-panel chart under `outputs/figures/`. The directories are created automatically if they don't exist.
```

