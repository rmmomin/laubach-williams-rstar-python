"""
Main entry point for LW 2023 replication.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .data import load_input_data
from .stages import stage1, stage2, stage3
from .utils import build_default_kappa_inputs


# Default constraints (matching R code)
A_R_CONSTRAINT = -0.0025
B_Y_CONSTRAINT = 0.025


def run_estimation(
    excel_path: str | Path,
    output_dir: str | Path = "outputs",
    use_kappa: bool = True,
    fix_phi: Optional[float] = None,
) -> Dict[str, float]:
    """
    Run the full LW 2023 replication and write CSV outputs.
    
    Parameters
    ----------
    excel_path : Path
        Path to Laubach_Williams_current_estimates.xlsx
    output_dir : Path
        Directory for output files
    use_kappa : bool
        Whether to use time-varying variance (kappa) for COVID period
    fix_phi : float, optional
        Fix phi at this value instead of estimating
    
    Returns
    -------
    Dict[str, float]
        Validation metrics vs published estimates
    """
    excel_path = Path(excel_path)
    outputs_path = Path(output_dir)
    data_path = outputs_path / "data"
    figures_path = outputs_path / "figures"
    data_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    prepared = load_input_data(excel_path)
    sample_end = (prepared.sample_end.year, prepared.sample_end.quarter)
    
    # Build kappa inputs
    kappa_inputs = None
    if use_kappa:
        kappa_inputs = build_default_kappa_inputs(
            (prepared.sample_start.year, prepared.sample_start.quarter)
        )
    
    print("Running Stage 1...")
    stage1_res = stage1(
        prepared,
        sample_end=sample_end,
        b_y_constraint=B_Y_CONSTRAINT,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs,
        fix_phi=fix_phi,
    )
    print(f"  lambda_g = {stage1_res.lambda_g:.6f}")
    
    print("Running Stage 2...")
    stage2_res = stage2(
        prepared,
        lambda_g=stage1_res.lambda_g,
        sample_end=sample_end,
        a_r_constraint=A_R_CONSTRAINT,
        b_y_constraint=B_Y_CONSTRAINT,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs,
        fix_phi=fix_phi,
    )
    print(f"  lambda_z = {stage2_res.lambda_z:.6f}")
    
    print("Running Stage 3...")
    stage3_res = stage3(
        prepared,
        lambda_g=stage1_res.lambda_g,
        lambda_z=stage2_res.lambda_z,
        sample_end=sample_end,
        a_r_constraint=A_R_CONSTRAINT,
        b_y_constraint=B_Y_CONSTRAINT,
        use_kappa=use_kappa,
        kappa_inputs=kappa_inputs,
        fix_phi=fix_phi,
    )
    print(f"  phi = {stage3_res.phi:.6f}")
    
    # Build results DataFrame
    results = pd.DataFrame({
        "Date": prepared.sample_periods.to_timestamp(),
        "rstar_filtered": stage3_res.rstar_filtered,
        "g_filtered": stage3_res.trend_filtered,
        "z_filtered": stage3_res.z_filtered,
        "output_gap_filtered": stage3_res.output_gap_filtered,
        "rstar_smoothed": stage3_res.rstar_smoothed,
        "g_smoothed": stage3_res.trend_smoothed,
        "z_smoothed": stage3_res.z_smoothed,
        "output_gap_smoothed": stage3_res.output_gap_smoothed,
    })
    
    csv_path = data_path / "lw_port_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Validate against published estimates and generate plots
    validation = _compare_with_reference(excel_path, results, figures_path)
    return validation


def _compare_with_reference(
    excel_path: Path, 
    results: pd.DataFrame,
    figures_path: Path,
) -> Dict[str, float]:
    """Compare results with published estimates from the Excel file and generate plots."""
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
        "rstar_filtered",
        "g_filtered",
        "z_filtered",
        "output_gap_filtered",
        "rstar_smoothed",
        "g_smoothed",
        "z_smoothed",
        "output_gap_smoothed",
    ]:
        ref_col = f"{col}_ref"
        if ref_col in merged.columns:
            diff = (merged[col] - merged[ref_col]).abs()
            metrics[f"{col}_max_abs_diff"] = diff.max()
            metrics[f"{col}_rmse"] = (diff**2).mean() ** 0.5
    
    # Generate comparison plots
    _generate_plots(merged, figures_path)
    
    return metrics


def _generate_plots(merged: pd.DataFrame, figures_path: Path) -> None:
    """Generate comparison and difference plots."""
    port_color = '#2E86AB'
    ref_color = '#E94F37'
    
    metrics = [
        ('rstar', 'r*', 'rstar_filtered', 'rstar_smoothed', 'rstar_filtered_ref', 'rstar_smoothed_ref'),
        ('g', 'g (trend growth)', 'g_filtered', 'g_smoothed', 'g_filtered_ref', 'g_smoothed_ref'),
        ('z', 'z (other factors)', 'z_filtered', 'z_smoothed', 'z_filtered_ref', 'z_smoothed_ref'),
        ('output_gap', 'Output Gap', 'output_gap_filtered', 'output_gap_smoothed', 'output_gap_filtered_ref', 'output_gap_smoothed_ref'),
    ]
    
    # Comparison plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('LW Python Port vs Published Estimates (2023 Replication)', fontsize=14, fontweight='bold')
    
    for row, (name, label, filt_port, smooth_port, filt_ref, smooth_ref) in enumerate(metrics):
        # Filtered
        ax = axes[row, 0]
        ax.plot(merged['Date'], merged[filt_port], color=port_color, linewidth=1.5, label='Python Port')
        ax.plot(merged['Date'], merged[filt_ref], color=ref_color, linewidth=1.5, linestyle='--', label='Published')
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} - Filtered', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Smoothed
        ax = axes[row, 1]
        ax.plot(merged['Date'], merged[smooth_port], color=port_color, linewidth=1.5, label='Python Port')
        ax.plot(merged['Date'], merged[smooth_ref], color=ref_color, linewidth=1.5, linestyle='--', label='Published')
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} - Smoothed', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    comparison_path = figures_path / 'lw_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {comparison_path}")
    
    # Difference plot
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 12))
    fig2.suptitle('Differences: Python Port - Published Estimates', fontsize=14, fontweight='bold')
    
    for row, (name, label, filt_port, smooth_port, filt_ref, smooth_ref) in enumerate(metrics):
        # Filtered diff
        ax = axes2[row, 0]
        diff = merged[filt_port] - merged[filt_ref]
        ax.plot(merged['Date'], diff, color='#4A4E69', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.fill_between(merged['Date'], diff, 0, alpha=0.3, color='#4A4E69')
        ax.set_ylabel(f'Δ {label}', fontsize=10)
        ax.set_title(f'{label} - Filtered Difference', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Smoothed diff
        ax = axes2[row, 1]
        diff = merged[smooth_port] - merged[smooth_ref]
        ax.plot(merged['Date'], diff, color='#4A4E69', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.fill_between(merged['Date'], diff, 0, alpha=0.3, color='#4A4E69')
        ax.set_ylabel(f'Δ {label}', fontsize=10)
        ax.set_title(f'{label} - Smoothed Difference', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    diff_path = figures_path / 'lw_differences.png'
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Differences plot saved to {diff_path}")
