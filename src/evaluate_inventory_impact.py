#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd


def evaluate_inventory_impact(
    predictions_path: str,
    output_dir: str,
    understock_cost: float = 5.0,
    overstock_cost: float = 1.0,
    buffer_grid: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)
    required_cols = {"category", "model", "actual", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if buffer_grid is None:
        buffer_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

    rows = []

    grouped = df.groupby(["model", "category"], as_index=False)
    for _, g in grouped:
        model = g["model"].iloc[0]
        category = int(g["category"].iloc[0])

        best_row = None
        for b in buffer_grid:
            order_qty = np.maximum(g["prediction"].values * (1.0 + b), 0.0)
            actual = np.maximum(g["actual"].values, 0.0)

            under = np.maximum(actual - order_qty, 0.0)
            over = np.maximum(order_qty - actual, 0.0)

            stockout_rate = float(np.mean(under > 0))
            service_level = float(np.mean(order_qty >= actual))
            mean_under = float(np.mean(under))
            mean_over = float(np.mean(over))

            weighted_cost = float(understock_cost * np.sum(under) + overstock_cost * np.sum(over))

            candidate = {
                "model": model,
                "category": category,
                "best_buffer": b,
                "service_level": service_level,
                "stockout_rate": stockout_rate,
                "mean_understock_units": mean_under,
                "mean_overstock_units": mean_over,
                "weighted_inventory_cost": weighted_cost,
            }

            if best_row is None or candidate["weighted_inventory_cost"] < best_row["weighted_inventory_cost"]:
                best_row = candidate

        rows.append(best_row)

    detailed_df = pd.DataFrame(rows)
    summary_df = (
        detailed_df.groupby("model", as_index=False)
        .agg(
            service_level_mean=("service_level", "mean"),
            stockout_rate_mean=("stockout_rate", "mean"),
            weighted_inventory_cost_mean=("weighted_inventory_cost", "mean"),
            categories_evaluated=("category", "nunique"),
        )
        .sort_values(["weighted_inventory_cost_mean", "stockout_rate_mean"], ascending=[True, True])
        .reset_index(drop=True)
    )

    os.makedirs(output_dir, exist_ok=True)
    detailed_out = os.path.join(output_dir, "inventory_impact_detailed.csv")
    summary_out = os.path.join(output_dir, "inventory_impact_summary.csv")
    detailed_df.to_csv(detailed_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    best = summary_df.iloc[0]
    print("\n✅ Inventory impact evaluation complete")
    print(f"Detailed results: {detailed_out}")
    print(f"Summary results:  {summary_out}")
    print(
        f"Best model by inventory cost: {best['model']} "
        f"(Cost={best['weighted_inventory_cost_mean']:.4f}, "
        f"Service={best['service_level_mean']:.3f}, Stockout={best['stockout_rate_mean']:.3f})"
    )

    return detailed_df, summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inventory planning impact from forecast predictions."
    )
    parser.add_argument(
        "--predictions-path",
        default="models/baseline_benchmark_predictions.csv",
        help="Path to benchmark prediction rows CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where inventory impact CSVs are written.",
    )
    parser.add_argument(
        "--understock-cost",
        type=float,
        default=5.0,
        help="Unit penalty for understock (lost-sales proxy).",
    )
    parser.add_argument(
        "--overstock-cost",
        type=float,
        default=1.0,
        help="Unit penalty for overstock (holding/waste proxy).",
    )
    args = parser.parse_args()

    evaluate_inventory_impact(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        understock_cost=args.understock_cost,
        overstock_cost=args.overstock_cost,
    )


if __name__ == "__main__":
    main()
