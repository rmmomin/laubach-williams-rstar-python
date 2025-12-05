from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .data import load_input_data
from .stages import stage1, stage2, stage3

AR_CONSTRAINT = -0.0025
BY_CONSTRAINT_STAGE2 = 0.025
BY_CONSTRAINT_STAGE3 = 0.025


def run_estimation(
    excel_path: str | Path,
    output_dir: str | Path = "outputs",
) -> Dict[str, float]:
    """
    Run the full LW replication stack and write CSV outputs for downstream use.

    Returns a dictionary with validation metrics versus the published estimates.
    """

    excel_path = Path(excel_path)
    outputs_path = Path(output_dir)
    data_path = outputs_path / "data"
    figures_path = outputs_path / "figures"
    data_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    prepared = load_input_data(excel_path)

    stage1_res = stage1(prepared)
    stage2_res = stage2(
        prepared,
        lambda_g=stage1_res.lambda_g,
        a_r_constraint=AR_CONSTRAINT,
        b_y_constraint=BY_CONSTRAINT_STAGE2,
    )
    stage3_res = stage3(
        prepared,
        lambda_g=stage1_res.lambda_g,
        lambda_z=stage2_res.lambda_z,
        a_r_constraint=AR_CONSTRAINT,
        b_y_constraint=BY_CONSTRAINT_STAGE3,
    )

    results = pd.DataFrame(
        {
            "Date": prepared.sample_periods.to_timestamp(),
            "rstar_filtered": stage3_res.rstar_filtered,
            "g_filtered": stage3_res.trend_filtered,
            "z_filtered": stage3_res.z_filtered,
            "output_gap_filtered": stage3_res.output_gap_filtered,
            "rstar_smoothed": stage3_res.rstar_smoothed,
            "g_smoothed": stage3_res.trend_smoothed,
            "z_smoothed": stage3_res.z_smoothed,
            "output_gap_smoothed": stage3_res.output_gap_smoothed,
        }
    )
    csv_path = data_path / "lw_port_results.csv"
    results.to_csv(csv_path, index=False)

    validation = _compare_with_reference(excel_path, results)
    return validation


def _compare_with_reference(excel_path: Path, results: pd.DataFrame) -> Dict[str, float]:
    ref = pd.read_excel(excel_path, sheet_name="data", skiprows=5)
    ref = ref.rename(
        columns={
            "rstar": "rstar_filtered_ref",
            "g": "g_filtered_ref",
            "z": "z_filtered_ref",
            "Output gap": "output_gap_filtered_ref",
            "rstar.1": "rstar_smoothed_ref",
            "g.1": "g_smoothed_ref",
            "z.1": "z_smoothed_ref",
            "Output gap.1": "output_gap_smoothed_ref",
        }
    )
    ref = ref.dropna(subset=["Date"]).copy()
    ref["Date"] = pd.to_datetime(ref["Date"])

    merged = results.merge(ref, on="Date", how="inner")

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
        diff = (merged[col] - merged[ref_col]).abs()
        metrics[f"{col}_max_abs_diff"] = diff.max()
        metrics[f"{col}_rmse"] = (diff**2).mean() ** 0.5
    return metrics

