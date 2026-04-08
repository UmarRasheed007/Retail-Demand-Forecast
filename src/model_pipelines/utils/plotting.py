#!/usr/bin/env python

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metrics_for_plot(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float, float]:
    yt = y_true.to_numpy(dtype=float)
    yp = y_pred.to_numpy(dtype=float)
    err = yt - yp

    rmse = float(np.sqrt(np.mean(np.square(err)))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    mean_error = float(np.mean(err)) if err.size else float("nan")

    nonzero_mask = yt != 0
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs(err[nonzero_mask] / yt[nonzero_mask])) * 100.0)
        accuracy = max(0.0, 100.0 - mape)
    else:
        accuracy = float("nan")

    return accuracy, mean_error, rmse, mae


def plot_category_forecast_grid(
    df_cmp: pd.DataFrame,
    model_name: str,
    out_path: str,
    max_categories: int = 6,
) -> None:
    if df_cmp.empty:
        return

    d = df_cmp.copy()
    d["dt"] = pd.to_datetime(d["dt"])
    cats = sorted(d["category"].astype(int).unique().tolist())[:max_categories]

    n = len(cats)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, cat in enumerate(cats):
        ax = axes[i]
        dc = d[d["category"].astype(int) == cat].sort_values("dt")
        # Collapse panel rows to one value per day for readable time-axis plots.
        daily = (
            dc.groupby("dt", as_index=False)[["actual", "prediction"]]
            .mean()
            .sort_values("dt")
        )
        ax.plot(daily["dt"], daily["actual"], marker="o", linewidth=2, label="Actual")
        ax.plot(daily["dt"], daily["prediction"], marker="o", linewidth=2, label="Predicted")

        accuracy, mean_error, metric_rmse, metric_mae = _metrics_for_plot(
            daily["actual"], daily["prediction"]
        )
        metrics_txt = (
            f"Accuracy: {accuracy:.2f}%\n"
            f"Error: {mean_error:.3f}\n"
            f"RMSE: {metric_rmse:.3f}\n"
            f"MAE: {metric_mae:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            metrics_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
        )

        ax.set_title(f"{model_name} - Category {cat}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Demand")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(summary_df: pd.DataFrame, metric_col: str, out_path: str, title: str) -> None:
    if summary_df.empty:
        return
    d = summary_df.sort_values(metric_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(d["model"], d[metric_col], color="#2f6db0")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_col)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histograms(pred_df: pd.DataFrame, out_path: str) -> None:
    if pred_df.empty:
        return
    models = sorted(pred_df["model"].unique().tolist())
    n = len(models)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, m in enumerate(models):
        ax = axes[i]
        d = pred_df[pred_df["model"] == m]
        ax.hist(d["error"].values, bins=25, alpha=0.85, color="#5b8bd2")
        ax.set_title(f"Residual Distribution - {m}")
        ax.set_xlabel("Error (actual - prediction)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
