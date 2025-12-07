# Laubach-Williams r* Python Port

A faithful Python port of the NY Fed's Laubach-Williams (2003) natural rate of interest (r*) model, based on the [2023 replication code](https://www.newyorkfed.org/research/policy/rstar).

The port replicates the three-stage estimation procedure:
1. **Stage 1**: Estimate potential output with trend growth
2. **Stage 2**: Add interest rate sensitivity, estimate λ_g (signal-to-noise ratio for trend growth)
3. **Stage 3**: Full model with r* = c·g + z, estimate λ_z (signal-to-noise ratio for z)

## Validation

All estimates match the published NY Fed figures within **0.04 percentage points**:

| Metric (Smoothed) | Max Abs Diff | RMSE |
|:------------------|-------------:|-----:|
| r* | 0.022 pp | 0.012 |
| g (trend growth) | 0.008 pp | 0.003 |
| z (other factors) | 0.011 pp | 0.009 |
| Output gap | 0.020 pp | 0.011 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download `Laubach_Williams_current_estimates.xlsx` from the [NY Fed website](https://www.newyorkfed.org/research/policy/rstar) and place it in `data/`.

### Updating Data (core inputs + relative price controls)
- Edit `data/Laubach_Williams_current_estimates.xlsx` (or your CSV equivalent) to append new quarters.
- Refresh core inputs:
  - Real GDP level (log): BEA NIPA real GDP (e.g., Table 1.1.6/1.1.3).
  - Core PCE price index: compute q/q annualized core PCE inflation (BEA Table 2.3.4/2.3.6).
    - The spreadsheet’s `inflation` column aligns with quarterly core PCE built from monthly `PCEPILFE` (average within quarter, 400·Δlog). `BPCCRO1Q156NBEA` is a close quarterly alternative but does not match as tightly.
  - Federal funds rate: H.15 monthly, averaged to quarterly.
  - Expected inflation: recompute 4-quarter-ahead expectation via AR(3) on core PCE over the extended sample (matches the spreadsheet logic). In checks, AR(4) on core PCEPILFE inflation fit the existing expectations best (RMSE ≈ 0.77 pp) versus SPF CPI 1Y (≈0.82) and SPF CPI 10Y (≈0.95). A 40-quarter rolling ARIMA(1,0,1) is similar unless you override the pandemic dips; overriding the 2021Q2–Q4 expectations to spreadsheet values improved RMSE to ~0.57 pp.
  - COVID dummy (`covid.ind`) and any kappa windows if you keep `use_kappa=True` (2020–2022 in the NY Fed setup; extend if you define new high-volatility periods).
- Refresh relative price controls used in the Phillips curve:
  - Crude oil import price inflation: q/q annualized log change of a petroleum import price series (BEA petroleum import price deflator or EIA refiner acquisition cost for imported crude).
  - Core import price inflation (nonpetroleum): q/q annualized log change of the import price deflator excluding petroleum (BEA) or BLS nonpetroleum import price index.
- After updating the file, rerun the pipeline (e.g., `python scripts/run_lw_port.py`).
- Handy codes (check vintage/definitions):
  - FRED: `GDPC1` (real GDP), `PCEPILFE` (core PCE price index), `FEDFUNDS` (federal funds), `RACIMUSDM` (refiners’ acquisition cost, imported crude oil).
  - FRED (core PCE quarterly alternative): `BPCCRO1Q156NBEA` (PCE excluding food & energy, chain-type price index).
  - BEA NIPA tables: 1.1.6/1.1.3 (real GDP), 2.3.4/2.3.6 (PCE price indexes), 4.2.4/4.2.6 (import price indexes for petroleum vs. nonpetroleum).
  - For nonpetroleum import prices via FRED/BLS, use the “Import Price Index: All Imports Excluding Petroleum” series (ticker varies by vintage; available in FRED search).

## Usage

### Modular Version (Recommended)

```bash
python scripts/run_lw_port.py
```

#### Inflation expectation diagnostics

```bash
python scripts/check_inflation_expectations.py
```

Generates AR(3)/AR(4) vs spreadsheet expectations, a 4-quarter moving-average vs spreadsheet comparison (CSV + PNG), SPF CPI 1Y/10Y comparisons (if `data/Inflation.xlsx` is present), ARIMA search (p=1..6, q=0..1), and the optional 2021Q2–Q4 override plot (includes MA(4)). Outputs land in `outputs/data/` and `outputs/figures/`.

### Condensed Single-File Version

For a standalone single-file version, use the `condensed/` directory:

```bash
cd condensed
python lw_estimation.py ../data/Laubach_Williams_current_estimates.xlsx
```

### Programmatic Usage

```python
from lwrep.run import run_estimation

metrics = run_estimation(
    excel_path="data/Laubach_Williams_current_estimates.xlsx",
    output_dir="outputs",
    sample_start=(1961, 1),
    sample_end=(2025, 2),
    use_kappa=True,  # COVID time-varying variance
)
```

## Project Structure

```
├── data/                   # Input data (Excel from NY Fed)
├── condensed/              # Single-file standalone version
│   └── lw_estimation.py    # All code in one file
├── outputs/
│   ├── data/               # CSV results
│   └── figures/            # Comparison plots
├── scripts/
│   └── run_lw_port.py      # Main entry point
├── src/lwrep/
│   ├── __init__.py
│   ├── kalman.py           # Kalman filter/smoother
│   ├── median_unbiased.py  # λ_g, λ_z estimation
│   ├── parameters.py       # State-space matrices
│   ├── run.py              # Orchestration & plotting
│   └── stages.py           # Stage 1, 2, 3 estimation
└── requirements.txt
```

## Output

Running the script produces:
- `outputs/data/lw_port_results.csv` - Filtered and smoothed estimates
- `outputs/figures/lw_comparison.png` - Time series comparison
- `outputs/figures/lw_differences.png` - Difference plots

## References

- Laubach, T., & Williams, J. C. (2003). Measuring the Natural Rate of Interest. *Review of Economics and Statistics*, 85(4), 1063-1070.
- [NY Fed r* Data and Code](https://www.newyorkfed.org/research/policy/rstar)
