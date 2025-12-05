from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


Quarter = pd.Period


def to_period(year: int, quarter: int) -> Quarter:
    """Return a pandas Period for the given year/quarter pair."""
    return pd.Period(freq="Q", year=year, quarter=quarter)


def shift_quarter(period: Quarter, shift: int) -> Quarter:
    """Mimic the R helper shiftQuarter()."""
    return period + shift


def r_slice(array: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Recreate R's 1-based inclusive slicing for 1-D numpy arrays.

    Parameters
    ----------
    array : np.ndarray
        Input 1-D vector.
    start : int
        1-based inclusive start index (matching R's convention).
    end : int
        1-based inclusive end index.
    """
    return array[start - 1 : end].copy()


def delayed_ramp(total_length: int, delay: int) -> np.ndarray:
    """
    Build the vector used for the polynomial detrending regressors in the R code.

    Equivalent to R's `c(rep(0, delay), 1:(total_length - delay))` when
    `total_length > delay`; otherwise returns zeros.
    """
    leading = min(delay, total_length)
    zeros = np.zeros(leading)
    remaining = total_length - leading
    if remaining <= 0:
        return zeros
    ramp = np.arange(1, remaining + 1, dtype=float)
    return np.concatenate([zeros, ramp])


@dataclass
class PreparedInput:
    log_output: np.ndarray
    inflation: np.ndarray
    rel_oil_inflation: np.ndarray
    rel_import_inflation: np.ndarray
    real_interest_rate: np.ndarray
    inflation_expectations: np.ndarray
    all_periods: pd.PeriodIndex
    sample_periods: pd.PeriodIndex
    sample_start: Quarter
    sample_end: Quarter

    @property
    def est_periods(self) -> pd.PeriodIndex:
        """Periods aligned with the state-space sample (i.e., drop first 8 pre-sample obs)."""
        return self.all_periods[8:]


def ensure_1d(array: Iterable[float]) -> np.ndarray:
    return np.asarray(array, dtype=float).reshape(-1)

