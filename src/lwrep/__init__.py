"""
Python port of the Laubach-Williams (2003) multi-stage r-star estimator.

This package mirrors the original R implementation found in `LW_replication/`
and exposes a high-level `run_estimation` helper that:

1. loads the published input data from `Laubach_Williams_current_estimates.xlsx`
2. runs the Stage 1–3 estimation procedure
3. writes filtered/smoothed outputs for comparison against the official release
"""

from .run import run_estimation

__all__ = ["run_estimation"]

