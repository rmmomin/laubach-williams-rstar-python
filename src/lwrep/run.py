"""
Main entry point - direct port of run.lw.R
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .median_unbiased import median_unbiased_estimator_stage1, median_unbiased_estimator_stage2
from .stages import rstar_stage1, rstar_stage2, rstar_stage3


def run_estimation(
    excel_path: str | Path,
    output_dir: str | Path = "outputs",
    sample_start: Tuple[int, int] = (1961, 1),
    sample_end: Tuple[int, int] = (2025, 2),
    use_kappa: bool = True,
    fix_phi: float | None = None,
) -> Dict[str, float]:
    """
    Run the full LW estimation - direct port of run.lw.R
    """
    excel_path = Path(excel_path)
    outputs_path = Path(output_dir)
    data_path = outputs_path / "data"
    figures_path = outputs_path / "figures"
    data_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Constraints from R code
    a_r_constraint = -0.0025
    b_y_constraint = 0.025
    
    # Load data
    data = pd.read_excel(excel_path, sheet_name="input data")
    
    # Find the row range based on sample dates
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Calculate data start (8 quarters before sample start)
    est_data_start_year = sample_start[0] - 2 if sample_start[1] == 1 else sample_start[0] - 1
    est_data_start_quarter = (sample_start[1] - 8 - 1) % 4 + 1
    if sample_start[1] <= 8:
        est_data_start_year = sample_start[0] - (9 - sample_start[1]) // 4 - 1
        est_data_start_quarter = (sample_start[1] - 8 - 1) % 4 + 1
    
    # Filter data
    start_date = pd.Timestamp(year=est_data_start_year, month=est_data_start_quarter * 3, day=1)
    end_date = pd.Timestamp(year=sample_end[0], month=sample_end[1] * 3, day=1)
    
    mask = (data["Date"] >= start_date) & (data["Date"] <= end_date)
    data = data[mask].copy().reset_index(drop=True)
    
    # Extract series
    log_output = data["gdp.log"].values
    inflation = data["inflation"].values
    relative_oil_price_inflation = data["oil.price.inflation"].values - inflation
    relative_import_price_inflation = data["import.price.inflation"].values - inflation
    nominal_interest_rate = data["interest"].values
    inflation_expectations = data["inflation.expectations"].values
    covid_dummy = data["covid.ind"].values
    real_interest_rate = nominal_interest_rate - inflation_expectations
    
    # Build kappa inputs
    kappa_inputs = None
    if use_kappa:
        kappa_inputs = pd.DataFrame({
            "name": ["kappa2020Q2-Q4", "kappa2021", "kappa2022"],
            "year": [2020, 2021, 2022],
            "T.start": [np.nan, np.nan, np.nan],
            "T.end": [np.nan, np.nan, np.nan],
            "init": [1.0, 1.0, 1.0],
            "lower.bound": [1.0, 1.0, 1.0],
            "upper.bound": [np.inf, np.inf, np.inf],
            "theta.index": [np.nan, np.nan, np.nan],
            "t.stat.null": [1.0, 1.0, 1.0],
        })
        
        # Calculate T.start and T.end
        for k in range(len(kappa_inputs)):
            year = kappa_inputs.loc[k, "year"]
            
            # T.start: index into y_t vector (which starts at sample_start)
            covid_variance_start = (year - sample_start[0]) * 4 + (1 - sample_start[1]) + 1
            kappa_inputs.loc[k, "T.start"] = max(covid_variance_start, 0)
            
            covid_variance_end = (year - sample_start[0]) * 4 + (4 - sample_start[1]) + 1
            kappa_inputs.loc[k, "T.end"] = max(covid_variance_end, 0)
            
            # Manual adjustment to start kappa_2020 in second quarter
            if year == 2020:
                kappa_inputs.loc[k, "T.start"] += 1
    
    print("=" * 60)
    print("Running Stage 1...")
    print("=" * 60)
    out_stage1 = rstar_stage1(
        log_output=log_output,
        inflation=inflation,
        relative_oil_price_inflation=relative_oil_price_inflation,
        relative_import_price_inflation=relative_import_price_inflation,
        covid_dummy=covid_dummy,
        sample_end=sample_end,
        b_y_constraint=b_y_constraint,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None,
        fix_phi=fix_phi,
    )
    
    # Median unbiased estimate of lambda_g
    lambda_g = median_unbiased_estimator_stage1(out_stage1.potential_smoothed)
    print(f"  lambda_g = {lambda_g:.6f}")
    
    print("=" * 60)
    print("Running Stage 2...")
    print("=" * 60)
    out_stage2 = rstar_stage2(
        log_output=log_output,
        inflation=inflation,
        relative_oil_price_inflation=relative_oil_price_inflation,
        relative_import_price_inflation=relative_import_price_inflation,
        real_interest_rate=real_interest_rate,
        covid_dummy=covid_dummy,
        lambda_g=lambda_g,
        sample_end=sample_end,
        a_r_constraint=a_r_constraint,
        b_y_constraint=b_y_constraint,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None,
        fix_phi=fix_phi,
    )
    
    # Median unbiased estimate of lambda_z
    lambda_z = median_unbiased_estimator_stage2(out_stage2.y, out_stage2.x, out_stage2.kappa_vec)
    print(f"  lambda_z = {lambda_z:.6f}")
    
    print("=" * 60)
    print("Running Stage 3...")
    print("=" * 60)
    out_stage3 = rstar_stage3(
        log_output=log_output,
        inflation=inflation,
        relative_oil_price_inflation=relative_oil_price_inflation,
        relative_import_price_inflation=relative_import_price_inflation,
        real_interest_rate=real_interest_rate,
        covid_dummy=covid_dummy,
        lambda_g=lambda_g,
        lambda_z=lambda_z,
        sample_end=sample_end,
        a_r_constraint=a_r_constraint,
        b_y_constraint=b_y_constraint,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None,
        fix_phi=fix_phi,
    )
    
    phi = out_stage3.theta[12]  # phi is at index 12 (param_num["phi"] - 1)
    print(f"  phi = {phi:.6f}")
    
    # Build output DataFrame
    t_end = len(log_output) - 8
    dates = pd.date_range(
        start=f"{sample_start[0]}-{sample_start[1]*3:02d}-01",
        periods=t_end,
        freq="QS"
    )
    
    results = pd.DataFrame({
        "Date": dates,
        "rstar_filtered": out_stage3.rstar_filtered,
        "g_filtered": out_stage3.trend_filtered,
        "z_filtered": out_stage3.z_filtered,
        "output_gap_filtered": out_stage3.output_gap_filtered,
        "rstar_smoothed": out_stage3.rstar_smoothed,
        "g_smoothed": out_stage3.trend_smoothed,
        "z_smoothed": out_stage3.z_smoothed,
        "output_gap_smoothed": out_stage3.output_gap_smoothed,
    })
    
    csv_path = data_path / "lw_port_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Compare with reference and generate plots
    validation = _compare_with_reference(excel_path, results, figures_path)
    
    return validation


def _compare_with_reference(
    excel_path: Path,
    results: pd.DataFrame,
    figures_path: Path,
) -> Dict[str, float]:
    """Compare with published estimates and generate plots."""
    ref = pd.read_excel(excel_path, sheet_name="data", skiprows=5)
    ref = ref.rename(columns={
        "rstar": "rstar_filtered_ref",
        "g": "g_filtered_ref",
        "z": "z_filtered_ref",
        "Output gap": "output_gap_filtered_ref",
        "rstar.1": "rstar_smoothed_ref",
        "g.1": "g_smoothed_ref",
        "z.1": "z_smoothed_ref",
        "Output gap.1": "output_gap_smoothed_ref",
    })
    ref = ref.dropna(subset=["Date"]).copy()
    ref["Date"] = pd.to_datetime(ref["Date"])
    
    merged = results.merge(ref, on="Date", how="inner")
    
    # Calculate metrics
    metrics = {}
    for col in [
        "rstar_filtered", "g_filtered", "z_filtered", "output_gap_filtered",
        "rstar_smoothed", "g_smoothed", "z_smoothed", "output_gap_smoothed",
    ]:
        ref_col = f"{col}_ref"
        if ref_col in merged.columns:
            diff = (merged[col] - merged[ref_col]).abs()
            metrics[f"{col}_max_abs_diff"] = diff.max()
            metrics[f"{col}_rmse"] = (diff ** 2).mean() ** 0.5
    
    # Generate plots
    _generate_plots(merged, figures_path)
    
    # Print validation
    print("\nValidation against published estimates:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    
    return metrics


def _generate_plots(merged: pd.DataFrame, figures_path: Path) -> None:
    """Generate comparison plots."""
    port_color = "#2E86AB"
    ref_color = "#E94F37"
    
    metrics = [
        ("rstar", "r*", "rstar_filtered", "rstar_smoothed", "rstar_filtered_ref", "rstar_smoothed_ref"),
        ("g", "g (trend growth)", "g_filtered", "g_smoothed", "g_filtered_ref", "g_smoothed_ref"),
        ("z", "z (other factors)", "z_filtered", "z_smoothed", "z_filtered_ref", "z_smoothed_ref"),
        ("output_gap", "Output Gap", "output_gap_filtered", "output_gap_smoothed", "output_gap_filtered_ref", "output_gap_smoothed_ref"),
    ]
    
    # Comparison plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("LW Python Port vs Published Estimates", fontsize=14, fontweight="bold")
    
    for row, (name, label, filt_port, smooth_port, filt_ref, smooth_ref) in enumerate(metrics):
        ax = axes[row, 0]
        ax.plot(merged["Date"], merged[filt_port], color=port_color, linewidth=1.5, label="Python Port")
        ax.plot(merged["Date"], merged[filt_ref], color=ref_color, linewidth=1.5, linestyle="--", label="Published")
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} - Filtered", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        
        ax = axes[row, 1]
        ax.plot(merged["Date"], merged[smooth_port], color=port_color, linewidth=1.5, label="Python Port")
        ax.plot(merged["Date"], merged[smooth_ref], color=ref_color, linewidth=1.5, linestyle="--", label="Published")
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} - Smoothed", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    comparison_path = figures_path / "lw_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {comparison_path}")
    
    # Difference plot
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 12))
    fig2.suptitle("Differences: Python Port - Published Estimates", fontsize=14, fontweight="bold")
    
    for row, (name, label, filt_port, smooth_port, filt_ref, smooth_ref) in enumerate(metrics):
        ax = axes2[row, 0]
        diff = merged[filt_port] - merged[filt_ref]
        ax.plot(merged["Date"], diff, color="#4A4E69", linewidth=1.5)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(merged["Date"], diff, 0, alpha=0.3, color="#4A4E69")
        ax.set_ylabel(f"Δ {label}", fontsize=10)
        ax.set_title(f"{label} - Filtered Difference", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        
        ax = axes2[row, 1]
        diff = merged[smooth_port] - merged[smooth_ref]
        ax.plot(merged["Date"], diff, color="#4A4E69", linewidth=1.5)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(merged["Date"], diff, 0, alpha=0.3, color="#4A4E69")
        ax.set_ylabel(f"Δ {label}", fontsize=10)
        ax.set_title(f"{label} - Smoothed Difference", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    diff_path = figures_path / "lw_differences.png"
    plt.savefig(diff_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Differences plot saved to {diff_path}")
