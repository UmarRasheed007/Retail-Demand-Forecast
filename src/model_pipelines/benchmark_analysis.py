#!/usr/bin/env python

"""Backward-compatible wrapper.

Use benchmark_analysis_publication.py for full publication-grade output.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_pipelines.benchmark_analysis_publication import main


if __name__ == "__main__":
    main()
