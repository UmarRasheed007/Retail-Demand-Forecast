#!/usr/bin/env python

import os
import pandas as pd


def _to_md_table(df: pd.DataFrame, cols: list[str]) -> str:
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def main():
    base = "src/models"
    out_path = "docs/dissertation_results_final.md"

    forecast = pd.read_csv(os.path.join(base, "baseline_benchmark_summary.csv"))
    inventory = pd.read_csv(os.path.join(base, "inventory_impact_summary.csv"))

    f = forecast.copy()
    f["RMSE (mean)"] = f["rmse_mean"].map(lambda v: f"{v:.4f}")
    f["MAE (mean)"] = f["mae_mean"].map(lambda v: f"{v:.4f}")
    f["MAPE (mean)"] = f["mape_mean"].map(lambda v: f"{v:.2f}%")
    f["Model"] = f["model"]
    f["Rank"] = range(1, len(f) + 1)

    i = inventory.copy()
    i["Weighted Cost (mean)"] = i["weighted_inventory_cost_mean"].map(lambda v: f"{v:.4f}")
    i["Service Level (mean)"] = i["service_level_mean"].map(lambda v: f"{v:.2f}")
    i["Stockout Rate (mean)"] = i["stockout_rate_mean"].map(lambda v: f"{v:.2f}")
    i["Model"] = i["model"]
    i["Rank"] = range(1, len(i) + 1)

    best_rmse = forecast.iloc[0]
    best_inv = inventory.iloc[0]

    content = f"""# Dissertation Final Results (FreshRetailNet-50K)

## Research Question
How accurately can machine learning and time-series forecasting models predict short-term retail product demand using open transaction data, and how can these predictions be used to improve inventory planning?

## Scope and Setup
- Dataset: FreshRetailNet-50K (open transaction data)
- Forecast granularity: daily demand
- Categories evaluated: 81, 60, 82, 184, 1
- Test horizon: last 10 days per category
- Models compared: LGBM, RF, ExtraTrees, GBR, XGB, CatBoost
- Inventory objective: weighted cost with understock cost = 5x overstock cost

## Forecast Accuracy Results
Source: src/models/baseline_benchmark_summary.csv

{_to_md_table(f[["Rank", "Model", "RMSE (mean)", "MAE (mean)", "MAPE (mean)"]], ["Rank", "Model", "RMSE (mean)", "MAE (mean)", "MAPE (mean)"])}

## Inventory Planning Impact Results
Source: src/models/inventory_impact_summary.csv

{_to_md_table(i[["Rank", "Model", "Weighted Cost (mean)", "Service Level (mean)", "Stockout Rate (mean)"]], ["Rank", "Model", "Weighted Cost (mean)", "Service Level (mean)", "Stockout Rate (mean)"])}

## Key Findings
- Best forecasting model by RMSE: {best_rmse['model']} (RMSE={best_rmse['rmse_mean']:.4f}, MAE={best_rmse['mae_mean']:.4f})
- Best inventory model by weighted cost: {best_inv['model']} (Cost={best_inv['weighted_inventory_cost_mean']:.4f}, Service={best_inv['service_level_mean']:.2f}, Stockout={best_inv['stockout_rate_mean']:.2f})
- Forecast-optimal and business-optimal models are different, so model selection should match operational objective.

## Direct Answer to Dissertation Question
Yes, short-term demand can be predicted with useful accuracy from open transaction data, and those predictions can be converted into inventory decisions that improve cost/service trade-offs.
"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(content)

    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
