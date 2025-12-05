# Laubach-Williams r* Python Port

A faithful Python port of the NY Fed's Laubach-Williams (2003) natural rate of interest (r*) model, based on the [2023 replication code](https://www.newyorkfed.org/research/policy/rstar).

The port replicates the three-stage estimation procedure:
1. **Stage 1**: Estimate potential output with trend growth
2. **Stage 2**: Add interest rate sensitivity, estimate λ_g (signal-to-noise ratio for trend growth)
3. **Stage 3**: Full model with r* = c·g + z, estimate λ_z (signal-to-noise ratio for z)

## Validation

All estimates match the published NY Fed figures within **0.04 percentage points**:

| Metric (Smoothed) | Max Abs Diff | RMSE |
|-------------------|--------------|------|
| r* | 0.022 pp | 0.012 |
| g (trend growth) | 0.008 pp | 0.003 |
| z (other factors) | 0.011 pp | 0.009 |
| Output gap | 0.020 pp | 0.011 |

## Setup

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Data

Download \`Laubach_Williams_current_estimates.xlsx\` from the [NY Fed website](https://www.newyorkfed.org/research/policy/rstar) and place it in \`data/\`.

## Usage

\`\`\`bash
python scripts/run_lw_port.py
\`\`\`

Or programmatically:

\`\`\`python
from lwrep.run import run_estimation

metrics = run_estimation(
    excel_path="data/Laubach_Williams_current_estimates.xlsx",
    output_dir="outputs",
    sample_start=(1961, 1),
    sample_end=(2025, 2),
    use_kappa=True,  # COVID time-varying variance
)
\`\`\`

## Project Structure

\`\`\`
├── data/                    # Input data (Excel from NY Fed)
├── outputs/
│   ├── data/                # CSV results
│   └── figures/             # Comparison plots
├── scripts/
│   └── run_lw_port.py       # Main entry point
├── src/lwrep/
│   ├── __init__.py
│   ├── kalman.py            # Kalman filter/smoother
│   ├── median_unbiased.py   # λ_g, λ_z estimation
│   ├── parameters.py        # State-space matrices
│   ├── run.py               # Orchestration & plotting
│   └── stages.py            # Stage 1, 2, 3 estimation
└── requirements.txt
\`\`\`

## Output

Running the script produces:
- \`outputs/data/lw_port_results.csv\` - Filtered and smoothed estimates
- \`outputs/figures/lw_comparison.png\` - Time series comparison
- \`outputs/figures/lw_differences.png\` - Difference plots

## References

- Laubach, T., & Williams, J. C. (2003). Measuring the Natural Rate of Interest. *Review of Economics and Statistics*, 85(4), 1063-1070.
- [NY Fed r* Data and Code](https://www.newyorkfed.org/research/policy/rstar)
