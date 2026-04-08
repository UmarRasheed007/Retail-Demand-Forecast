#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_pipelines.nbeats.core import train_nbeats
from model_pipelines.utils.data import resolve_data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train N-BEATS per category")
    parser.add_argument("--cats", nargs="+", type=int, required=True)
    parser.add_argument("--modelready-path", default="data/daily_dataset/daily_df_modelready.parquet")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--input-len", type=int, default=28)
    parser.add_argument("--output-len", type=int, default=7)
    args = parser.parse_args()

    args.modelready_path = resolve_data_path(args.modelready_path, anchor_file=__file__)
    args.model_dir = resolve_data_path(args.model_dir, anchor_file=__file__)

    train_nbeats(
        categories=args.cats,
        modelready_path=args.modelready_path,
        model_dir=args.model_dir,
        input_len=args.input_len,
        output_len=args.output_len,
    )


if __name__ == "__main__":
    main()
