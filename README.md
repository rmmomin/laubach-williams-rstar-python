# LW Python Port

This repository exists to build a Python port of the Laubach-Williams (2003) natural-rate model using Bayesian state-space estimation. The core implementation lives in `lw_rstar_bayesian_state_space.py`, which combines custom Kalman filtering with PyMC priors that mirror the original literature.

## Environment setup

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

> Dependencies are pinned to versions that install cleanly on Python 3.9. If you upgrade to Python 3.10+ you can relax the caps to track the latest PyMC/PyTensor/pandas releases.

Once the dependencies are installed you can swap the synthetic data in `main()` for your own FRED-prepped dataset and run the estimator end-to-end.

