#!/usr/bin/env python3
"""
Run the LW Python port estimation.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lwrep.run import run_estimation

def main():
    project_root = Path(__file__).parent.parent
    excel_path = project_root / "data" / "Laubach_Williams_current_estimates.xlsx"
    
    metrics = run_estimation(
        excel_path=excel_path,
        output_dir=project_root / "outputs",
        sample_start=(1961, 1),
        sample_end=(2025, 2),
        use_kappa=True,
        fix_phi=None,
    )
    return metrics


if __name__ == "__main__":
    main()
