# LW Python Port

This repository houses a Python port of the Laubach-Williams (2003) natural-rate model based on the official `LW_replication/` R code. The port mirrors the three-stage estimation (median-unbiased signal-to-noise ratios + Kalman filtering/smoothing) and uses the published `Laubach_Williams_current_estimates.xlsx` file for both input data and validation.

## Environment setup

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

> Dependencies are pinned to versions that install cleanly on Python 3.9. If you upgrade to Python 3.10+ you can relax the caps to track the latest PyMC/PyTensor/pandas releases.

### Run the synthetic demo

`python scripts/run_lw_port.py`

### Use your own data

Import `run_estimation` from `lwrep.run` if you need programmatic access, or run the script above to generate the filtered/smoothed estimates plus a comparison against the published New York Fed figures.

## Project layout

```
├── requirements.txt
├── LW_replication/              # Original R code + Excel data
├── src/
│   └── lwrep/
│       ├── __init__.py
│       ├── data.py
│       ├── kalman.py
│       ├── parameters.py
│       ├── run.py
│       └── stages.py
├── outputs/
│   ├── data/
│   └── figures/
└── scripts/
    └── run_lw_port.py

Running `python scripts/run_lw_port.py` produces `outputs/data/lw_port_results.csv` (filtered and smoothed series) and prints RMS/max-abs differences versus the official spreadsheet so you can confirm the port stays aligned.
```

