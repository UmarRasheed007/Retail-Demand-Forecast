#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_pipelines.baselines.core import run_baseline_train_predict, generate_baseline_figures
from model_pipelines.utils.data import prepare_modelready_if_missing

MODEL_KEY = "rf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with RF and generate model-specific PNG")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--batch-size", type=int, default=12000)
    parser.add_argument("--flat-dir", default="data/flattened_chunks_eval")
    parser.add_argument("--daily-path", default="data/daily_dataset/daily_df_eval.parquet")
    parser.add_argument("--modelready-path", default="data/daily_dataset/daily_df_eval_modelready.parquet")
    parser.add_argument("--cats", nargs="+", type=int, required=True)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--max-rows-per-category", type=int, default=None)
    parser.add_argument("--output-dir", default="src/models/benchmark_artifacts/rf")
    args = parser.parse_args()

    prepare_modelready_if_missing(
        modelready_path=args.modelready_path,
        split=args.split,
        batch_size=args.batch_size,
        flat_dir=args.flat_dir,
        daily_path=args.daily_path,
    )

    _, summary_df, pred_df = run_baseline_train_predict(
        model_key=MODEL_KEY,
        modelready_path=args.modelready_path,
        categories=args.cats,
        test_days=args.test_days,
        output_dir=args.output_dir,
        max_rows_per_category=args.max_rows_per_category,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pred_out = os.path.join(args.output_dir, "predictions_rf.csv")
    pred_df.to_csv(pred_out, index=False)
    generate_baseline_figures(MODEL_KEY, pred_df, summary_df, args.output_dir)


if __name__ == "__main__":
    main()
