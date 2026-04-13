#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_inventory_impact import evaluate_inventory_impact
from model_pipelines.baselines.core import (
    run_baseline_train_predict,
    build_pred_with_errors,
    generate_baseline_figures,
)
from model_pipelines.nbeats.core import predict_nbeats, generate_nbeats_figure
from model_pipelines.utils.data import resolve_data_path
from model_pipelines.utils.metrics import rmse, mae, safe_mape
from model_pipelines.utils.plotting import (
    plot_metric_bars,
    plot_residual_histograms,
)

PROJECT_TITLE = "Data-Driven Retail Demand Forecasting Using Machine Learning for Inventory Optimisation"
RESEARCH_QUESTION = (
    "How accurately can machine learning and time-series forecasting models predict short-term "
    "retail product demand using open transaction data, and how can these predictions be used "
    "to improve inventory planning?"
)


def _to_md_table(df: pd.DataFrame, cols: list[str]) -> str:
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def _compute_per_category_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, category), g in pred_df.groupby(["model", "category"]):
        rows.append(
            {
                "model": model,
                "category": int(category),
                "rmse": rmse(g["actual"].values, g["prediction"].values),
                "mae": mae(g["actual"].values, g["prediction"].values),
                "mape": safe_mape(g["actual"].values, g["prediction"].values),
            }
        )
    return pd.DataFrame(rows)


def _compute_summary(per_cat: pd.DataFrame) -> pd.DataFrame:
    return (
        per_cat.groupby("model", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mape_mean=("mape", "mean"),
            categories_evaluated=("category", "nunique"),
        )
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )


def _build_mean_ensemble(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(columns=["category", "model", "dt", "actual", "prediction", "error", "abs_error"])

    d = pred_df.copy()
    d["dt"] = pd.to_datetime(d["dt"])

    # Take actual from first occurrence, average predictions across models for each category/date.
    ens = (
        d.groupby(["category", "dt"], dropna=False, as_index=False)
        .agg(
            actual=("actual", "first"),
            prediction=("prediction", "mean"),
        )
        .reset_index(drop=True)
    )
    ens["model"] = "ensemble_mean"
    ens["error"] = ens["actual"] - ens["prediction"]
    ens["abs_error"] = np.abs(ens["error"])
    return ens[["category", "model", "dt", "actual", "prediction", "error", "abs_error"]]


def _build_inventory_scenarios(pred_path: str, out_dir: str) -> pd.DataFrame:
    scenarios = [(5.0, 1.0), (3.0, 1.0), (1.0, 1.0)]
    rows = []
    for under_cost, over_cost in scenarios:
        scen_dir = os.path.join(out_dir, f"inventory_u{int(under_cost)}_o{int(over_cost)}")
        _, summary = evaluate_inventory_impact(
            predictions_path=pred_path,
            output_dir=scen_dir,
            understock_cost=under_cost,
            overstock_cost=over_cost,
        )
        summary = summary.copy()
        summary["understock_cost"] = under_cost
        summary["overstock_cost"] = over_cost
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _write_publication_report(
    report_path: str,
    cats: Iterable[int],
    test_days: int,
    summary_df: pd.DataFrame,
    inventory_scenarios_df: pd.DataFrame,
    artifacts_dir: str,
) -> None:
    disp = summary_df.copy()
    disp["Rank"] = np.arange(1, len(disp) + 1)
    disp["Model"] = disp["model"]
    disp["RMSE"] = disp["rmse_mean"].map(lambda v: f"{v:.4f}")
    disp["MAE"] = disp["mae_mean"].map(lambda v: f"{v:.4f}")
    disp["MAPE"] = disp["mape_mean"].map(lambda v: f"{v:.2f}%")

    inv_disp = inventory_scenarios_df.copy()
    inv_disp["Scenario"] = inv_disp.apply(
        lambda r: f"under={r['understock_cost']:.0f}, over={r['overstock_cost']:.0f}", axis=1
    )
    inv_disp["Model"] = inv_disp["model"]
    inv_disp["Weighted Cost"] = inv_disp["weighted_inventory_cost_mean"].map(lambda v: f"{v:.4f}")
    inv_disp["Service Level"] = inv_disp["service_level_mean"].map(lambda v: f"{v:.3f}")
    inv_disp["Stockout Rate"] = inv_disp["stockout_rate_mean"].map(lambda v: f"{v:.3f}")

    best = summary_df.iloc[0]

    md = f"""# Publication Benchmark Report

## Project Title
{PROJECT_TITLE}

## Research Question
{RESEARCH_QUESTION}

## Experimental Setup
- Categories evaluated: {', '.join(map(str, cats))}
- Forecast horizon: {test_days} days
- Models benchmarked: {', '.join(summary_df['model'].tolist())}

## Forecast Accuracy Leaderboard
{_to_md_table(disp[["Rank", "Model", "RMSE", "MAE", "MAPE", "categories_evaluated"]], ["Rank", "Model", "RMSE", "MAE", "MAPE", "categories_evaluated"])}

## Inventory Scenarios
{_to_md_table(inv_disp[["Scenario", "Model", "Weighted Cost", "Service Level", "Stockout Rate", "categories_evaluated"]], ["Scenario", "Model", "Weighted Cost", "Service Level", "Stockout Rate", "categories_evaluated"])}

## Key Finding
Best forecasting model by RMSE: {best['model']} (RMSE={best['rmse_mean']:.4f}, MAE={best['mae_mean']:.4f}, MAPE={best['mape_mean']:.2f}%)

## Artifact Location
{artifacts_dir}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full benchmark across N-BEATS and classical ML models."
    )
    parser.add_argument("--train-modelready-path", default="data/daily_dataset/daily_df_modelready.parquet")
    parser.add_argument("--eval-modelready-path", default="data/daily_dataset/daily_df_eval_modelready.parquet")
    parser.add_argument("--cats", nargs="+", type=int, default=[81, 60, 82, 184, 1])
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        default=["lgbm", "rf", "extra_trees", "gbr", "xgb", "catboost"],
    )
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--max-rows-per-category", type=int, default=None)
    parser.add_argument("--nbeats-model-dir", default="src/models")
    parser.add_argument("--nbeats-input-len", type=int, default=28)
    parser.add_argument("--nbeats-output-len", type=int, default=7)
    parser.add_argument("--output-dir", default="src/models/publication_benchmark")
    args = parser.parse_args()

    args.train_modelready_path = resolve_data_path(args.train_modelready_path, anchor_file=__file__)
    args.eval_modelready_path = resolve_data_path(args.eval_modelready_path, anchor_file=__file__)
    args.nbeats_model_dir = resolve_data_path(args.nbeats_model_dir, anchor_file=__file__)

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    all_pred_parts = []

    # Baseline models: train + predict + per-model figures and files.
    for model_key in args.baseline_models:
        model_out = os.path.join(args.output_dir, model_key)
        os.makedirs(model_out, exist_ok=True)

        _, summary_df, pred_df = run_baseline_train_predict(
            model_key=model_key,
            modelready_path=args.train_modelready_path,
            categories=args.cats,
            test_days=args.test_days,
            output_dir=model_out,
            max_rows_per_category=args.max_rows_per_category,
        )
        pred_clean = build_pred_with_errors(pred_df, model_key)
        pred_clean.to_csv(os.path.join(model_out, f"predictions_{model_key}.csv"), index=False)
        generate_baseline_figures(model_key, pred_clean, summary_df, model_out)
        all_pred_parts.append(pred_clean)

    # N-BEATS predictions + figure.
    nbeats_pred = predict_nbeats(
        train_modelready_path=args.train_modelready_path,
        eval_modelready_path=args.eval_modelready_path,
        categories=args.cats,
        model_dir=args.nbeats_model_dir,
        input_len=args.nbeats_input_len,
        output_len=args.nbeats_output_len,
    )

    if not nbeats_pred.empty:
        nbeats_out_dir = os.path.join(args.output_dir, "nbeats")
        os.makedirs(nbeats_out_dir, exist_ok=True)
        nbeats_pred.to_csv(os.path.join(nbeats_out_dir, "predictions_nbeats.csv"), index=False)
        generate_nbeats_figure(nbeats_pred, os.path.join(nbeats_out_dir, "forecast_nbeats_category_grid.png"))
        all_pred_parts.append(nbeats_pred)

    if not all_pred_parts:
        raise RuntimeError("No predictions generated. Check input paths and model files.")

    full_pred = pd.concat(all_pred_parts, ignore_index=True)
    full_pred["dt"] = pd.to_datetime(full_pred["dt"])

    ensemble_pred = _build_mean_ensemble(full_pred)
    if not ensemble_pred.empty:
        full_pred = pd.concat([full_pred, ensemble_pred], ignore_index=True)

    full_pred_path = os.path.join(args.output_dir, "full_benchmark_predictions.csv")
    full_pred.to_csv(full_pred_path, index=False)

    per_cat = _compute_per_category_metrics(full_pred)
    per_cat_path = os.path.join(args.output_dir, "full_benchmark_per_category_metrics.csv")
    per_cat.to_csv(per_cat_path, index=False)

    summary = _compute_summary(per_cat)
    summary_path = os.path.join(args.output_dir, "full_benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Global comparison figures with robust naming.
    plot_metric_bars(
        summary,
        metric_col="rmse_mean",
        out_path=os.path.join(figures_dir, "comparison_rmse_all_models.png"),
        title="RMSE Comparison Across Models",
    )
    plot_metric_bars(
        summary,
        metric_col="mae_mean",
        out_path=os.path.join(figures_dir, "comparison_mae_all_models.png"),
        title="MAE Comparison Across Models",
    )
    plot_metric_bars(
        summary,
        metric_col="mape_mean",
        out_path=os.path.join(figures_dir, "comparison_mape_all_models.png"),
        title="MAPE Comparison Across Models",
    )

    plot_residual_histograms(
        full_pred,
        out_path=os.path.join(figures_dir, "diagnostic_residual_histograms_all_models.png"),
    )

    inventory_scenarios = _build_inventory_scenarios(full_pred_path, os.path.join(args.output_dir, "inventory_scenarios"))
    inventory_scenarios_path = os.path.join(args.output_dir, "inventory_scenarios_summary.csv")
    inventory_scenarios.to_csv(inventory_scenarios_path, index=False)

    report_path = os.path.join(args.output_dir, "publication_benchmark_report.md")
    _write_publication_report(
        report_path=report_path,
        cats=args.cats,
        test_days=args.test_days,
        summary_df=summary,
        inventory_scenarios_df=inventory_scenarios,
        artifacts_dir=args.output_dir,
    )

    print("Publication benchmark artifacts generated")
    print(f"Predictions: {full_pred_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
