#!/usr/bin/env python

import os
from typing import Iterable

import numpy as np
import pandas as pd

from train_baseline_benchmarks import benchmark_models_for_categories
from model_pipelines.utils.plotting import plot_category_forecast_grid


def run_baseline_train_predict(
    model_key: str,
    modelready_path: str,
    categories: Iterable[int],
    test_days: int,
    output_dir: str,
    max_rows_per_category: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train and predict for one baseline model via shared benchmark engine."""
    os.makedirs(output_dir, exist_ok=True)

    detailed_df, summary_df = benchmark_models_for_categories(
        modelready_path=modelready_path,
        cats=list(categories),
        model_names=[model_key],
        output_dir=output_dir,
        test_days=test_days,
        max_rows_per_category=max_rows_per_category,
    )

    pred_path = os.path.join(output_dir, "baseline_benchmark_predictions.csv")
    pred_df = pd.read_csv(pred_path)
    pred_df["dt"] = pd.to_datetime(pred_df["dt"])
    pred_df = pred_df[pred_df["model"] == model_key].copy()
    return detailed_df, summary_df, pred_df


def generate_baseline_figures(
    model_key: str,
    pred_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures_dir: str,
) -> dict[str, str]:
    os.makedirs(figures_dir, exist_ok=True)

    compare_png = os.path.join(figures_dir, f"forecast_{model_key}_category_grid.png")
    plot_category_forecast_grid(pred_df, model_key.upper(), compare_png)

    out = {
        "category_grid": compare_png,
    }

    # Single-row summary with explicit naming for publication traceability.
    summary_out = os.path.join(figures_dir, f"metrics_{model_key}_summary.csv")
    summary_df.assign(model=model_key).to_csv(summary_out, index=False)
    out["metrics_csv"] = summary_out
    return out


def build_pred_with_errors(pred_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    d = pred_df.copy()
    d["model"] = model_key
    if "error" not in d.columns:
        d["error"] = d["actual"] - d["prediction"]
    if "abs_error" not in d.columns:
        d["abs_error"] = np.abs(d["error"])
    return d[["category", "model", "dt", "actual", "prediction", "error", "abs_error"]]
