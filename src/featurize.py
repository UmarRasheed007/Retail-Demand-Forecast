#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def build_features(input_path: str, output_path: str) -> None:
    """
    Reads daily_df_imputed.parquet, creates lags, rolling, calendar features,
    drops NaNs, and writes out a model-ready Parquet.
    """
    df = pd.read_parquet(input_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["third_category_id", "dt"])

    # calendar
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["dt"].dt.day
    df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
    df["month"] = df["dt"].dt.month
    df["quarter"] = df["dt"].dt.quarter

    # cyclical encodings for seasonality
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    # relative time index
    df["time_idx"] = (df["dt"] - df["dt"].min()).dt.days

    # lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = (
            df.groupby("third_category_id")["daily_sale_imputed"]
              .shift(lag)
        )

    # rolling statistics and trend/smoothing signals
    g = df.groupby("third_category_id")["daily_sale_imputed"]
    shifted = g.shift(1)
    for w in [3, 7, 14, 28]:
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = shifted.rolling(w).std().reset_index(level=0, drop=True)
        df[f"roll_min_{w}"] = shifted.rolling(w).min().reset_index(level=0, drop=True)
        df[f"roll_max_{w}"] = shifted.rolling(w).max().reset_index(level=0, drop=True)

    df["ewm_mean_7"] = shifted.ewm(span=7, adjust=False).mean().reset_index(level=0, drop=True)
    df["ewm_mean_14"] = shifted.ewm(span=14, adjust=False).mean().reset_index(level=0, drop=True)

    # momentum-style features
    df["diff_1"] = df["lag_1"] - df["lag_2"]
    df["diff_7"] = df["lag_1"] - df["lag_7"]
    df["ratio_7_28"] = (df["roll_mean_7"] / (df["roll_mean_28"] + 1e-6)).replace([np.inf, -np.inf], np.nan)

    # drop initial NaNs
    df_model = df.replace([np.inf, -np.inf], np.nan).dropna()

    # write out
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_model.to_parquet(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Build time-series features on daily imputed dataset."
    )
    parser.add_argument(
        "--input-path", default="src/data/daily_dataset/daily_df_imputed.parquet",
        help="Path to the aggregated+imputed daily Parquet."
    )
    parser.add_argument(
        "--output-path", default="src/data/daily_dataset/daily_df_modelready.parquet",
        help="Path for the model-ready Parquet."
    )
    args = parser.parse_args()
    build_features(args.input_path, args.output_path)

if __name__ == "__main__":
    main()