"""
Python port of the Laubach-Williams (2003) r* estimation model.

This package faithfully replicates the 2023 R code from the NY Fed.
"""

from .run import run_estimation

__all__ = ["run_estimation"]
