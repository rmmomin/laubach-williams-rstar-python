#!/usr/bin/env python3
"""
Compare r* estimates under different inflation expectation proxies:
- Reported (spreadsheet)
- 4-quarter moving average of inflation
- ARIMA(1,0,1) expectation (from diagnostics output)
- SPF CPI 1Y (if available in diagnostics output)

Outputs:
- CSV: outputs/data/rstar_expectations_comparison.csv
- Figure: outputs/figures/rstar_expectations_comparison.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lwrep.median_unbiased import median_unbiased_estimator_stage1, median_unbiased_estimator_stage2
from lwrep.stages import rstar_stage1, rstar_stage2, rstar_stage3

DATA_DIR = PROJECT_ROOT / "data"
OUT_DATA = PROJECT_ROOT / "outputs" / "data"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"
EXCEL_PATH = DATA_DIR / "Laubach_Williams_current_estimates.xlsx"

SAMPLE_START = (1961, 1)
SAMPLE_END = (2025, 2)


def load_input_sheet() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, sheet_name="input data")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_expectations(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build expectation variants, filling missing values with reported expectations."""
    reported = df["inflation.expectations"]

    # MA(4) of quarterly inflation; backfill first 3 with reported to avoid NaNs
    ma4 = df["inflation"].rolling(4).mean()
    ma4.iloc[:3] = reported.iloc[:3]

    # ARIMA(1,0,1) and SPF CPI 1Y from diagnostics output if present
    arima_path = OUT_DATA / "inflation_expectations_arima101_override.csv"
    arima_series = pd.Series(np.nan, index=df.index)
    spf_series = pd.Series(np.nan, index=df.index)
    if arima_path.exists():
        arima_df = pd.read_csv(arima_path, parse_dates=["date"])
        arima_merge = df[["Date"]].merge(
            arima_df[["date", "exp.arima101", "spf.cpi1y"]].rename(columns={"date": "Date"}),
            on="Date",
            how="left",
        )
        arima_series = arima_merge["exp.arima101"]
        spf_series = arima_merge["spf.cpi1y"]

    arima_filled = arima_series.fillna(reported)
    spf_filled = spf_series.fillna(reported)

    return {
        "reported": reported,
        "ma4": ma4,
        "arima101": arima_filled,
        "spf_cpi1y": spf_filled,
    }


def run_pipeline(
    df: pd.DataFrame,
    expectations: pd.Series,
    sample_start: tuple[int, int] = SAMPLE_START,
    sample_end: tuple[int, int] = SAMPLE_END,
    use_kappa: bool = True,
    fix_phi: float | None = None,
) -> pd.DataFrame:
    """Run the three-stage LW pipeline for a given expectations series."""
    df = df.copy()
    df["inflation.expectations"] = expectations.values

    # Match date handling from run_estimation
    est_data_start_year = sample_start[0] - 2 if sample_start[1] == 1 else sample_start[0] - 1
    est_data_start_quarter = (sample_start[1] - 8 - 1) % 4 + 1
    if sample_start[1] <= 8:
        est_data_start_year = sample_start[0] - (9 - sample_start[1]) // 4 - 1
        est_data_start_quarter = (sample_start[1] - 8 - 1) % 4 + 1

    start_date = pd.Timestamp(year=est_data_start_year, month=est_data_start_quarter * 3, day=1)
    end_date = pd.Timestamp(year=sample_end[0], month=sample_end[1] * 3, day=1)
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    data = df.loc[mask].reset_index(drop=True)

    inflation = data["inflation"].to_numpy()
    log_output = data["gdp.log"].to_numpy()
    rel_oil = data["oil.price.inflation"].to_numpy() - inflation
    rel_import = data["import.price.inflation"].to_numpy() - inflation
    nominal_rate = data["interest"].to_numpy()
    inflation_exp = data["inflation.expectations"].to_numpy()
    covid_dummy = data["covid.ind"].to_numpy()
    real_rate = nominal_rate - inflation_exp

    # Kappa inputs (COVID variances)
    kappa_inputs = None
    if use_kappa:
        kappa_inputs = pd.DataFrame(
            {
                "name": ["kappa2020Q2-Q4", "kappa2021", "kappa2022"],
                "year": [2020, 2021, 2022],
                "T.start": [np.nan, np.nan, np.nan],
                "T.end": [np.nan, np.nan, np.nan],
                "init": [1.0, 1.0, 1.0],
                "lower.bound": [1.0, 1.0, 1.0],
                "upper.bound": [np.inf, np.inf, np.inf],
                "theta.index": [np.nan, np.nan, np.nan],
                "t.stat.null": [1.0, 1.0, 1.0],
            }
        )
        for k in range(len(kappa_inputs)):
            year = kappa_inputs.loc[k, "year"]
            covid_variance_start = (year - sample_start[0]) * 4 + (1 - sample_start[1]) + 1
            kappa_inputs.loc[k, "T.start"] = max(covid_variance_start, 0)
            covid_variance_end = (year - sample_start[0]) * 4 + (4 - sample_start[1]) + 1
            kappa_inputs.loc[k, "T.end"] = max(covid_variance_end, 0)
            if year == 2020:
                kappa_inputs.loc[k, "T.start"] += 1

    # Stage 1
    out_stage1 = rstar_stage1(
        log_output=log_output,
        inflation=inflation,
        relative_oil_price_inflation=rel_oil,
        relative_import_price_inflation=rel_import,
        covid_dummy=covid_dummy,
        sample_end=sample_end,
        b_y_constraint=0.025,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None,
        fix_phi=fix_phi,
    )
    lambda_g = median_unbiased_estimator_stage1(out_stage1.potential_smoothed)

    # Stage 2
    out_stage2 = rstar_stage2(
        log_output=log_output,
        inflation=inflation,
        relative_oil_price_inflation=rel_oil,
        relative_import_price_inflation=rel_import,
        real_interest_rate=real_rate,
        covid_dummy=covid_dummy,
        lambda_g=lambda_g,
        sample_end=sample_end,
        a_r_constraint=-0.0025,
        b_y_constraint=0.025,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None,
        fix_phi=fix_phi,
    )
    lambda_z = median_unbiased_estimator_stage2(out_stage2.y, out_stage2.x, out_stage2.kappa_vec)

    # Stage 3
    out_stage3 = rstar_stage3(
        log_output=log_output,
        inflation=inflation,
        relative_oil_price_inflation=rel_oil,
        relative_import_price_inflation=rel_import,
        real_interest_rate=real_rate,
        covid_dummy=covid_dummy,
        lambda_g=lambda_g,
        lambda_z=lambda_z,
        sample_end=sample_end,
        a_r_constraint=-0.0025,
        b_y_constraint=0.025,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None,
        fix_phi=fix_phi,
    )

    t_end = len(log_output) - 8
    dates = pd.date_range(
        start=f"{sample_start[0]}-{sample_start[1]*3:02d}-01",
        periods=t_end,
        freq="QS",
    )
    return pd.DataFrame({"Date": dates, "rstar_smoothed": out_stage3.rstar_smoothed})


def main():
    warnings.filterwarnings("ignore")
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    input_df = load_input_sheet()
    expectations = prepare_expectations(input_df)

    results = []
    for name, series in expectations.items():
        res = run_pipeline(input_df, series)
        res = res.rename(columns={"rstar_smoothed": f"rstar_{name}"})
        results.append(res)

    merged = results[0]
    for res in results[1:]:
        merged = merged.merge(res, on="Date", how="inner")

    out_csv = OUT_DATA / "rstar_expectations_comparison.csv"
    merged.to_csv(out_csv, index=False)

    plt.figure(figsize=(12, 6))
    for col, color in zip(
        ["rstar_reported", "rstar_ma4", "rstar_arima101", "rstar_spf_cpi1y"],
        ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
    ):
        plt.plot(merged["Date"], merged[col], label=col.replace("rstar_", "").upper(), linewidth=1.6, color=color)
    plt.title("r* estimates under alternative inflation expectations")
    plt.ylabel("Percent (annual)")
    plt.xlabel("Date")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig = OUT_FIGS / "rstar_expectations_comparison.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print(f"Saved comparison data to {out_csv}")
    print(f"Saved comparison figure to {out_fig}")


if __name__ == "__main__":
    main()

