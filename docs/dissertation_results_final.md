# Dissertation Final Results (FreshRetailNet-50K)

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

| Rank | Model | RMSE (mean) | MAE (mean) | MAPE (mean) |
|---|---|---:|---:|---:|
| 1 | GBR | 1.1711 | 0.8269 | 40.48% |
| 2 | XGB | 1.1966 | 0.8164 | 41.79% |
| 3 | ExtraTrees | 1.2395 | 0.9191 | 47.83% |
| 4 | LGBM | 1.2586 | 0.8632 | 39.90% |
| 5 | CatBoost | 1.2742 | 0.9168 | 46.58% |
| 6 | RF | 1.3408 | 0.9206 | 44.12% |

### Accuracy Interpretation
- Best pure forecasting accuracy by RMSE is GBR.
- XGB is very close to GBR and has the best MAE among the top-2 models.
- LGBM has better mean MAPE than most models but weaker RMSE than GBR/XGB.

## Inventory Planning Impact Results
Source: src/models/inventory_impact_summary.csv

| Rank | Model | Weighted Cost (mean) | Service Level (mean) | Stockout Rate (mean) |
|---|---|---:|---:|---:|
| 1 | XGB | 12.7178 | 0.66 | 0.34 |
| 2 | LGBM | 13.5126 | 0.70 | 0.30 |
| 3 | RF | 14.1809 | 0.70 | 0.30 |
| 4 | CatBoost | 15.0705 | 0.72 | 0.28 |
| 5 | GBR | 15.5994 | 0.66 | 0.34 |
| 6 | ExtraTrees | 15.8978 | 0.60 | 0.40 |

### Inventory Interpretation
- Best inventory-cost model is XGB, not GBR.
- This shows model selection depends on objective:
  - forecasting objective (RMSE): GBR
  - inventory objective (cost under asymmetric risk): XGB
- For practical operations, inventory-aware model selection is better than accuracy-only selection.

## Direct Answer to Dissertation Question
Yes, short-term demand can be predicted with useful accuracy from open transaction data.

- Forecasting side: RMSE around 1.17 to 1.34 across tested models on selected categories demonstrates stable predictive signal.
- Inventory side: Using forecasts to set order quantities and optimize safety buffer materially changes business outcomes; XGB achieved the lowest weighted inventory cost under understock-heavy penalties.

Therefore, the project demonstrates both:
1) measurable demand-forecasting accuracy, and
2) actionable inventory-planning improvements from forecast-driven ordering policy.

## Reproducibility Commands
Run from project root.

1. Build image:
   docker compose build

2. Run final-stage evaluation:
   docker compose run --rm --remove-orphans trainer sh -lc "python src/run_final_stage.py"

3. Output files:
- src/models/baseline_benchmark_summary.csv
- src/models/baseline_benchmark_detailed.csv
- src/models/baseline_benchmark_predictions.csv
- src/models/inventory_impact_summary.csv
- src/models/inventory_impact_detailed.csv

## Notes for Dissertation Write-up
- Current final-stage run uses a per-category row cap (default 2000 rows) for Docker stability.
- In dissertation text, state this as an execution constraint and list full-scale rerun as future validation.
- The methodological conclusion remains valid: best model by prediction error is not always best by inventory business objective.
