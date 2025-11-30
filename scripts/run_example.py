#!/usr/bin/env python3
"""
Convenience entry point for running the synthetic-data demo.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from laubach_williams_rstar.model import main  # noqa: E402


if __name__ == "__main__":
    main()

