#!/usr/bin/env python3
"""
CLI for running the Python port of the Laubach-Williams replication code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lwrep.run import run_estimation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LW replication port.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("LW_replication/Laubach_Williams_current_estimates.xlsx"),
        help="Path to Laubach_Williams_current_estimates.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV outputs.",
    )
    args = parser.parse_args()

    metrics = run_estimation(args.excel, args.output_dir)
    print("Validation against published estimates:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

