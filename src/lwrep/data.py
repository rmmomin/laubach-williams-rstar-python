from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import PreparedInput, ensure_1d, shift_quarter, to_period


DEFAULT_SAMPLE_START = (1961, 1)
DEFAULT_SAMPLE_END = (2019, 2)
PRE_SAMPLE_QUARTERS = 8


def load_input_data(
    excel_path: Path,
    sample_start: Tuple[int, int] = DEFAULT_SAMPLE_START,
    sample_end: Tuple[int, int] = DEFAULT_SAMPLE_END,
) -> PreparedInput:
    """
    Load the published LW input data directly from the replication Excel.

    Parameters
    ----------
    excel_path : Path
        Path to `Laubach_Williams_current_estimates.xlsx`.
    sample_start, sample_end : Tuple[int, int]
        Year/quarter tuples describing the estimation window.
    """

    df = pd.read_excel(excel_path, sheet_name="input data")
    df["Date"] = pd.PeriodIndex(df["Date"], freq="Q")

    sample_start_period = to_period(*sample_start)
    sample_end_period = to_period(*sample_end)
    est_start_period = shift_quarter(sample_start_period, -PRE_SAMPLE_QUARTERS)

    mask = (df["Date"] >= est_start_period) & (df["Date"] <= sample_end_period)
    trimmed = df.loc[mask].copy()
    if trimmed.empty:
        raise ValueError(
            f"No data found between {est_start_period} and {sample_end_period} "
            f"in {excel_path}"
        )

    inflation = ensure_1d(trimmed["inflation"])
    rel_oil = ensure_1d(trimmed["oil.price.inflation"] - trimmed["inflation"])
    rel_import = ensure_1d(trimmed["import.price.inflation"] - trimmed["inflation"])
    log_output = ensure_1d(trimmed["gdp.log"])
    inflation_exp = ensure_1d(trimmed["inflation.expectations"])
    real_rate = ensure_1d(trimmed["interest"] - trimmed["inflation.expectations"])

    all_periods = pd.PeriodIndex(trimmed["Date"], freq="Q")
    sample_periods = all_periods[PRE_SAMPLE_QUARTERS:]

    return PreparedInput(
        log_output=log_output,
        inflation=inflation,
        rel_oil_inflation=rel_oil,
        rel_import_inflation=rel_import,
        real_interest_rate=real_rate,
        inflation_expectations=inflation_exp,
        all_periods=all_periods,
        sample_periods=sample_periods,
        sample_start=sample_start_period,
        sample_end=sample_end_period,
    )

