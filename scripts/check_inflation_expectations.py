"""
Diagnostics for the inflation-expectation series used in the LW pipeline.

This script:
- Reconstructs quarterly core PCE inflation from monthly PCEPILFE.
- Compares AR(3)/AR(4) expectations to the spreadsheet expectations.
- Compares model expectations to SPF CPI 1Y and 10Y.
- Searches over ARIMA(p,0,q) (p=1..6, q=0..1) for best fit.
- Applies a pandemic override for 2021Q2–Q4 to match the spreadsheet dip.

Outputs:
- CSVs in outputs/data/
- Figures in outputs/figures/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DATA = PROJECT_ROOT / "outputs" / "data"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"


def load_input_sheet() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "Laubach_Williams_current_estimates.xlsx", sheet_name="input data")
    df["date"] = pd.to_datetime(df["Date"])
    return df


def core_pcepilfe_quarterly() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "PCEPILFE.csv", parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "PCEPILFE": "pcepilfe"}).sort_values("date")
    q = (
        df.assign(q=df["date"].dt.to_period("Q"))
        .groupby("q")["pcepilfe"]
        .mean()
        .reset_index()
    )
    q["date"] = q["q"].dt.to_timestamp(how="start")
    q = q[["date", "pcepilfe"]].sort_values("date")
    q["inflation.pcepilfe"] = 400 * np.log(q["pcepilfe"]).diff()
    # Year-over-year quarterly inflation (not annualized) for 4Q moving average
    q["inflation.pcepilfe.yoy"] = 100 * np.log(q["pcepilfe"]).diff(4)
    return q


def spf_cpi() -> pd.DataFrame:
    path = DATA_DIR / "Inflation.xlsx"
    if not path.exists():
        return pd.DataFrame(columns=["date", "INFCPI1YR", "INFCPI10YR"])
    df = pd.read_excel(path, sheet_name="INFLATION")
    df["date"] = pd.PeriodIndex(year=df["YEAR"], quarter=df["QUARTER"], freq="Q").to_timestamp(how="start")
    return df[["date", "INFCPI1YR", "INFCPI10YR"]]


def ar_expectations(series: pd.Series, order: int, window: int = 40, horizon: int = 4) -> pd.Series:
    """Rolling AR(p) with intercept; fit on last `window` obs, forecast `horizon`, average."""
    y = series.to_numpy(dtype=float)
    n = len(y)
    res = np.full(n, np.nan)
    p = order
    for t in range(n):
        if t < max(p, window):
            continue
        sample = y[t - window : t]
        if np.isnan(sample).any():
            continue
        Y = sample[p:]
        if len(Y) == 0:
            continue
        X = [np.ones(len(Y))]
        for k in range(p):
            X.append(sample[p - 1 - k : -1 - k])
        X = np.column_stack(X)
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        c, phis = beta[0], beta[1:]
        lags = list(y[t - p : t][::-1])
        forecasts = []
        prev = lags.copy()
        for _ in range(horizon):
            f = c
            for k, phi in enumerate(phis):
                f += phi * (prev[k] if k < len(prev) else forecasts[-(k - len(prev) + 1)])
            forecasts.append(f)
            prev = [f] + prev
        res[t] = float(np.mean(forecasts))
    return pd.Series(res, index=series.index)


def arima_search(
    infl: np.ndarray, exp_model: np.ndarray, orders: Iterable[Tuple[int, int, int]], window: int = 40, horizon: int = 4
) -> Tuple[pd.DataFrame, dict]:
    results = []
    forecasts_store = {}
    for order in orders:
        p, d, q = order
        fc = np.full(len(infl), np.nan)
        for t in range(len(infl)):
            if t < max(window, p + d + q + 1):
                continue
            sample = infl[t - window : t]
            if np.isnan(sample).any():
                continue
            try:
                fit = ARIMA(sample, order=order).fit()
                f = fit.forecast(steps=horizon)
                fc[t] = float(f.mean())
            except Exception:
                continue
        rmse = np.sqrt(np.nanmean((fc - exp_model) ** 2))
        valid = int(np.sum(~np.isnan(fc) & ~np.isnan(exp_model)))
        results.append({"order": order, "rmse": rmse, "valid": valid})
        forecasts_store[order] = fc
    res_df = pd.DataFrame(results).sort_values("rmse")
    return res_df, forecasts_store


def rolling_arima_forecast(series: pd.Series, order: Tuple[int, int, int], window: int = 40, horizon: int = 4) -> pd.Series:
    """Rolling ARIMA forecast; mean of horizon steps."""
    y = series.to_numpy(dtype=float)
    n = len(y)
    res = np.full(n, np.nan)
    for t in range(n):
        if t < max(window, order[0] + order[1] + order[2] + 1):
            continue
        sample = y[t - window : t]
        if np.isnan(sample).any():
            continue
        try:
            fit = ARIMA(sample, order=order).fit()
            f = fit.forecast(steps=horizon)
            res[t] = float(f.mean())
        except Exception:
            continue
    return pd.Series(res, index=series.index)


def plot_series(dates, series_list, labels, title, path):
    plt.figure(figsize=(10, 5))
    for s, lbl in zip(series_list, labels):
        plt.plot(dates, s, label=lbl)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Percent (annual)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    warnings.filterwarnings("ignore")
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    input_df = load_input_sheet()
    model_exp = input_df["inflation.expectations"]
    dates = input_df["date"]

    core_q = core_pcepilfe_quarterly()
    merged = input_df[["date", "inflation.expectations", "inflation"]].merge(
        core_q[["date", "inflation.pcepilfe", "inflation.pcepilfe.yoy"]], on="date", how="left"
    )

    # 4-quarter simple moving average of quarterly inflation (q/q annualized) to mirror the sheet proxy
    merged["exp.ma4"] = merged["inflation"].rolling(4).mean()

    # 4-quarter moving average of year-over-year inflation (distinct from MA(4) above)
    merged["exp.ma4.yoy"] = merged["inflation.pcepilfe.yoy"].rolling(4).mean()
    merged["exp.ma4.yoy"] = merged["exp.ma4.yoy"].fillna(merged["inflation.expectations"])

    # Rolling MA(4) model (Box-Jenkins) on quarterly inflation (p=0, d=0, q=4)
    merged["exp.ma4.model"] = rolling_arima_forecast(merged["inflation.pcepilfe"], order=(0, 0, 4), window=40, horizon=4)

    merged[
        [
            "date",
            "inflation",
            "inflation.pcepilfe.yoy",
            "inflation.expectations",
            "exp.ma4",
            "exp.ma4.yoy",
            "exp.ma4.model",
        ]
    ].to_csv(OUT_DATA / "inflation_expectations_ma4.csv", index=False)
    plot_series(
        dates,
        [model_exp, merged["exp.ma4"], merged["exp.ma4.yoy"]],
        [
            "Model exp (sheet)",
            "4Q moving avg of q/q annualized inflation (current + 3 lags)",
            "4Q moving avg of YoY inflation",
        ],
        "Inflation expectations: sheet vs 4Q moving averages",
        OUT_FIGS / "inflation_expectations_ma4.png",
    )
    # Simpler view: just sheet vs q/q 4Q moving average
    plot_series(
        dates,
        [model_exp, merged["exp.ma4"]],
        ["Model exp (sheet)", "4Q moving avg of q/q annualized inflation (current + 3 lags)"],
        "Inflation expectations: sheet vs 4Q moving average (q/q annualized)",
        OUT_FIGS / "inflation_expectations_ma4_qoq_only.png",
    )

    # AR(3) vs AR(4) on core PCEPILFE
    merged["exp.ar3.pcepilfe"] = ar_expectations(merged["inflation.pcepilfe"], order=3, window=40)
    merged["exp.ar4.pcepilfe"] = ar_expectations(merged["inflation.pcepilfe"], order=4, window=40)
    merged.to_csv(OUT_DATA / "inflation_expectations_ar3_ar4_pcepilfe.csv", index=False)

    plot_series(
        dates,
        [model_exp, merged["exp.ar3.pcepilfe"], merged["exp.ar4.pcepilfe"]],
        ["Model exp (sheet)", "AR(3) PCEPILFE", "AR(4) PCEPILFE"],
        "Inflation expectations: AR(3) vs AR(4) from PCEPILFE",
        OUT_FIGS / "inflation_expectations_ar3_ar4_pcepilfe.png",
    )

    # SPF comparison
    spf = spf_cpi()
    merged_spf = merged.merge(spf, on="date", how="left")
    merged_spf.to_csv(OUT_DATA / "inflation_expectations_spf_comparison.csv", index=False)

    plot_series(
        dates,
        [model_exp, merged["exp.ar4.pcepilfe"], merged_spf["INFCPI1YR"], merged_spf["INFCPI10YR"]],
        ["Model exp (sheet)", "AR(4) PCEPILFE", "SPF CPI 1Y", "SPF CPI 10Y"],
        "Inflation expectations: AR(4) vs SPF CPI 1Y / 10Y",
        OUT_FIGS / "inflation_expectations_ar4_vs_spf_cpi1y.png",
    )

    # ARIMA search (p=1..6, q=0..1, d=0)
    mask = ~merged["inflation.pcepilfe"].isna() & ~model_exp.isna()
    infl = merged.loc[mask, "inflation.pcepilfe"].to_numpy()
    exp_model = model_exp[mask].to_numpy()
    dates_masked = dates[mask]

    orders = [(p, 0, q) for p in range(1, 7) for q in range(0, 2)]
    res_df, forecasts_store = arima_search(infl, exp_model, orders, window=40, horizon=4)
    res_df.to_csv(OUT_DATA / "arima_search_results.csv", index=False)

    best_order = tuple(res_df.iloc[0]["order"])
    best_fc = forecasts_store[best_order]
    pd.DataFrame(
        {"date": dates_masked, "exp.model": exp_model, "exp.best_arima": best_fc, "inflation.pcepilfe": infl}
    ).to_csv(OUT_DATA / "inflation_expectations_best_arima.csv", index=False)

    plot_series(
        dates_masked,
        [exp_model, best_fc],
        [ "Model exp (sheet)", f"Best ARIMA{best_order}" ],
        f"Inflation expectations: best ARIMA vs model (window=40)",
        OUT_FIGS / "inflation_expectations_best_arima.png",
    )

    # ARIMA(1,0,1) with pandemic overrides for 2021Q2–Q4 (plot with SPF CPI 1Y)
    arima101_fc = forecasts_store.get((1, 0, 1))
    if arima101_fc is None:
        arima101_fc = np.full(len(infl), np.nan)
    overrides = {
        pd.Timestamp("2021-04-01"): 0.165344642386979,
        pd.Timestamp("2021-07-01"): -0.880504730092122,
        pd.Timestamp("2021-10-01"): -2.32804322737388,
    }
    arima101_override = arima101_fc.copy()
    for dt, val in overrides.items():
        idx = np.where(dates_masked == dt)[0]
        if len(idx) == 1:
            arima101_override[idx[0]] = val

    spf_cpi1y_masked = merged_spf.loc[mask, "INFCPI1YR"].to_numpy() if "mask" in locals() else np.full(len(infl), np.nan)

    pd.DataFrame(
        {
            "date": dates_masked,
            "exp.model": exp_model,
            "exp.arima101": arima101_fc,
            "exp.arima101.override": arima101_override,
            "spf.cpi1y": spf_cpi1y_masked,
        }
    ).to_csv(OUT_DATA / "inflation_expectations_arima101_override.csv", index=False)

    exp_ma4_masked = merged.loc[mask, "exp.ma4"].to_numpy()
    plot_series(
        dates_masked,
        [exp_model, exp_ma4_masked, merged.loc[mask, "exp.ma4.model"].to_numpy(), arima101_fc, arima101_override, spf_cpi1y_masked],
        [
            "Model exp (sheet)",
            "4Q moving avg of q/q annualized inflation",
            "MA(4) model forecast (rolling ARIMA 0,0,4)",
            "ARIMA(1,0,1)",
            "ARIMA(1,0,1) + override 2021Q2-4",
            "SPF CPI 1Y",
        ],
        "Inflation expectations: sheet vs MA filters/models and ARIMA/SPF",
        OUT_FIGS / "inflation_expectations_arima101_override.png",
    )

    # Print summary to console
    rmse_ar3 = np.sqrt(np.nanmean((merged["exp.ar3.pcepilfe"] - model_exp) ** 2))
    rmse_ar4 = np.sqrt(np.nanmean((merged["exp.ar4.pcepilfe"] - model_exp) ** 2))
    rmse_ma4 = np.sqrt(np.nanmean((merged["exp.ma4"] - model_exp) ** 2))
    rmse_ma4_yoy = np.sqrt(np.nanmean((merged["exp.ma4.yoy"] - model_exp) ** 2))
    rmse_ma4_model = np.sqrt(np.nanmean((merged["exp.ma4.model"] - model_exp) ** 2))
    rmse_best = np.sqrt(np.nanmean((best_fc - exp_model) ** 2))
    rmse_over = np.sqrt(np.nanmean((arima101_override - exp_model) ** 2))
    print(f"MA(4) RMSE vs model: {rmse_ma4:.3f}")
    print(f"4Q MA YoY RMSE vs model: {rmse_ma4_yoy:.3f}")
    print(f"MA(4) model RMSE vs model: {rmse_ma4_model:.3f}")
    print(f"AR(3) RMSE vs model: {rmse_ar3:.3f}")
    print(f"AR(4) RMSE vs model: {rmse_ar4:.3f}")
    print(f"Best ARIMA{best_order} RMSE vs model: {rmse_best:.3f}")
    print(f"ARIMA(1,0,1) + override RMSE vs model: {rmse_over:.3f}")


if __name__ == "__main__":
    main()

