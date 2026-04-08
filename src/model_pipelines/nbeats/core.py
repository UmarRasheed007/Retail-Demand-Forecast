#!/usr/bin/env python

import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries

from train_darts_nbeats import load_nbeats_model, main as train_nbeats_main
from model_pipelines.utils.metrics import rmse, mae, safe_mape
from model_pipelines.utils.plotting import plot_category_forecast_grid


def train_nbeats(
    categories: Iterable[int],
    modelready_path: str,
    model_dir: str,
    input_len: int,
    output_len: int,
) -> None:
    train_nbeats_main(
        cats=list(categories),
        modelready_path=modelready_path,
        model_dir=model_dir,
        input_len=input_len,
        output_len=output_len,
    )


def predict_nbeats(
    train_modelready_path: str,
    eval_modelready_path: str,
    categories: Iterable[int],
    model_dir: str,
    input_len: int,
    output_len: int,
) -> pd.DataFrame:
    df_train = pd.read_parquet(train_modelready_path)
    df_eval = pd.read_parquet(eval_modelready_path)
    df_train["dt"] = pd.to_datetime(df_train["dt"])
    df_eval["dt"] = pd.to_datetime(df_eval["dt"])

    rows = []

    pl_kwargs = {"precision": 32}
    if torch.backends.mps.is_available():
        pl_kwargs.update({"accelerator": "mps", "devices": 1})

    for cat in categories:
        df_tr = (
            df_train[df_train["third_category_id"] == cat]
            .groupby("dt", as_index=False)["daily_sale_imputed"]
            .sum()
            .sort_values("dt")
        )
        df_ev = (
            df_eval[df_eval["third_category_id"] == cat]
            .groupby("dt", as_index=False)["daily_sale_imputed"]
            .sum()
            .sort_values("dt")
        )
        if df_tr.empty or df_ev.empty:
            continue

        ts_tr = TimeSeries.from_dataframe(
            df_tr,
            time_col="dt",
            value_cols="daily_sale_imputed",
            freq="D",
        ).astype(np.float32)

        history = ts_tr[-input_len:]

        weights_path = os.path.join(model_dir, f"nbeats_cat_{cat}.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model file not found: {weights_path}")

        model = load_nbeats_model(
            path=weights_path,
            input_chunk_length=input_len,
            output_chunk_length=output_len,
            random_state=42,
            pl_trainer_kwargs=pl_kwargs,
        )

        pred_ts = model.predict(n=output_len, series=history)
        pred_df = pred_ts.to_dataframe().reset_index()
        pred_df.columns = ["dt", "prediction"]
        pred_df["category"] = int(cat)

        cmp = pd.merge(pred_df, df_ev, on="dt", how="left").rename(
            columns={"daily_sale_imputed": "actual"}
        )
        cmp["model"] = "nbeats"
        cmp["error"] = cmp["actual"] - cmp["prediction"]
        cmp["abs_error"] = np.abs(cmp["error"])
        rows.extend(cmp[["category", "model", "dt", "actual", "prediction", "error", "abs_error"]].to_dict("records"))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["dt"] = pd.to_datetime(out["dt"])
    return out


def summarize_nbeats(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(columns=["model", "rmse_mean", "mae_mean", "mape_mean", "categories_evaluated"])

    rows = []
    for cat, g in pred_df.groupby("category"):
        rows.append(
            {
                "category": int(cat),
                "rmse": rmse(g["actual"].values, g["prediction"].values),
                "mae": mae(g["actual"].values, g["prediction"].values),
                "mape": safe_mape(g["actual"].values, g["prediction"].values),
            }
        )

    d = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "model": "nbeats",
                "rmse_mean": float(d["rmse"].mean()),
                "mae_mean": float(d["mae"].mean()),
                "mape_mean": float(d["mape"].mean()),
                "categories_evaluated": int(d["category"].nunique()),
            }
        ]
    )


def generate_nbeats_figure(pred_df: pd.DataFrame, out_path: str) -> None:
    plot_category_forecast_grid(pred_df, "NBEATS", out_path)
