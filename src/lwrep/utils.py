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
    covid_indicator: np.ndarray
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


@dataclass
class KappaConfig:
    """Configuration for time-varying variance (kappa) during COVID period."""
    name: str
    year: int
    T_start: int
    T_end: int
    init: float = 1.0
    lower_bound: float = 1.0
    upper_bound: float = float("inf")
    theta_index: int = -1


def build_default_kappa_inputs(sample_start: Tuple[int, int]) -> list:
    """
    Build default kappa inputs for 2020-2022 COVID variance scaling.
    Matches R code kappa.inputs data.frame.
    """
    kappas = []
    for year in [2020, 2021, 2022]:
        # Calculate T.start and T.end relative to sample_start
        start_offset = (year - sample_start[0]) * 4 + (1 - sample_start[1])
        T_start = max(start_offset + 1, 0)
        T_end = max(start_offset + 4, 0)
        
        # Manual adjustment: kappa_2020 starts in Q2
        if year == 2020:
            T_start += 1
        
        kappas.append(KappaConfig(
            name=f"kappa{year}Q2-Q4" if year == 2020 else f"kappa{year}",
            year=year,
            T_start=T_start,
            T_end=T_end,
        ))
    return kappas


def hp_filter(y: np.ndarray, lamb: float = 36000.0) -> np.ndarray:
    """
    Hodrick-Prescott filter for trend extraction.
    
    Parameters
    ----------
    y : np.ndarray
        Time series to filter.
    lamb : float
        Smoothing parameter (default 36000 for quarterly data, matching R code).
    
    Returns
    -------
    np.ndarray
        Trend component.
    """
    n = len(y)
    # Build the second difference matrix
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    
    # Solve (I + lamb * D'D) * trend = y
    I = np.eye(n)
    A = I + lamb * (D.T @ D)
    trend = np.linalg.solve(A, y)
    return trend

