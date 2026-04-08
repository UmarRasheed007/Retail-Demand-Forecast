#!/usr/bin/env python

import sys
from pathlib import Path


def ensure_src_on_path(anchor_file: str) -> Path:
    """Add src directory to sys.path when running scripts directly."""
    src_dir = Path(anchor_file).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir
