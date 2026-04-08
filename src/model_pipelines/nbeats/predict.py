#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_pipelines.nbeats.core import predict_nbeats, generate_nbeats_figure
from model_pipelines.utils.data import prepare_modelready_if_missing, resolve_data_path


    args.modelready_path = resolve_data_path(args.modelready_path, anchor_file=__file__)
    args.train_modelready_path = resolve_data_path(args.train_modelready_path, anchor_file=__file__)
    args.daily_path = resolve_data_path(args.daily_path, anchor_file=__file__)
    args.flat_dir = resolve_data_path(args.flat_dir, anchor_file=__file__)
    args.model_dir = resolve_data_path(args.model_dir, anchor_file=__file__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with N-BEATS and generate model-specific PNG")
    parser.add_argument("--batch-size", type=int, default=12000)
    parser.add_argument("--flat-dir", default="data/flattened_chunks_eval")
    parser.add_argument("--daily-path", default="data/daily_dataset/daily_df_eval.parquet")
    parser.add_argument("--modelready-path", default="data/daily_dataset/daily_df_eval_modelready.parquet")
    parser.add_argument("--train-modelready-path", required=True)
    parser.add_argument("--cats", nargs="+", type=int, required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--input-len", type=int, default=28)
    parser.add_argument("--output-len", type=int, default=7)
    parser.add_argument("--output-dir", default="src/models/benchmark_artifacts/nbeats")
    args = parser.parse_args()

    prepare_modelready_if_missing(
        modelready_path=args.modelready_path,
        split="eval",
        batch_size=args.batch_size,
        flat_dir=args.flat_dir,
        daily_path=args.daily_path,
    )

    pred_df = predict_nbeats(
        train_modelready_path=args.train_modelready_path,
        eval_modelready_path=args.modelready_path,
        categories=args.cats,
        model_dir=args.model_dir,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pred_out = os.path.join(args.output_dir, "predictions_nbeats.csv")
    fig_out = os.path.join(args.output_dir, "forecast_nbeats_category_grid.png")
    pred_df.to_csv(pred_out, index=False)
    generate_nbeats_figure(pred_df, fig_out)

    print(f"Saved: {pred_out}")
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
