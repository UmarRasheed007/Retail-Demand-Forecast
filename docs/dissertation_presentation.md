---
marp: true
theme: default
paginate: true
---

# Stockout-Aware Retail Demand Forecasting
## Final Dissertation Results

- Dataset: FreshRetailNet-50K
- Goal: short-term demand forecasting + inventory planning impact
- Candidate models: LGBM, RF, ExtraTrees, GBR, XGB, CatBoost

---

# Problem Statement
How accurately can ML and time-series models predict short-term retail demand from open transaction data, and how can predictions improve inventory planning?

---

# Pipeline Built
1. Ingest and flatten hourly records
2. Aggregate to daily and impute latent demand
3. Feature engineering (lags, rolling, calendar)
4. Multi-model benchmark
5. Inventory impact simulation with asymmetric stockout/overstock penalties

---

# Experimental Setup
- Categories: 81, 60, 82, 184, 1
- Test window: last 10 days/category
- Inventory penalty: understock = 5x overstock
- Outputs:
  - Forecast metrics: RMSE, MAE, MAPE
  - Inventory metrics: weighted cost, service level, stockout rate

---

# Forecast Accuracy Ranking

| Rank | Model | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|
| 1 | GBR | 1.1711 | 0.8269 | 40.48% |
| 2 | XGB | 1.1966 | 0.8164 | 41.79% |
| 3 | ExtraTrees | 1.2395 | 0.9191 | 47.83% |
| 4 | LGBM | 1.2586 | 0.8632 | 39.90% |
| 5 | CatBoost | 1.2742 | 0.9168 | 46.58% |
| 6 | RF | 1.3408 | 0.9206 | 44.12% |

---

# Inventory Planning Ranking

| Rank | Model | Weighted Cost | Service | Stockout |
|---|---|---:|---:|---:|
| 1 | XGB | 12.7178 | 0.66 | 0.34 |
| 2 | LGBM | 13.5126 | 0.70 | 0.30 |
| 3 | RF | 14.1809 | 0.70 | 0.30 |
| 4 | CatBoost | 15.0705 | 0.72 | 0.28 |
| 5 | GBR | 15.5994 | 0.66 | 0.34 |
| 6 | ExtraTrees | 15.8978 | 0.60 | 0.40 |

---

# Key Insight
Best model depends on objective:

- Best forecasting accuracy (RMSE): **GBR**
- Best inventory business outcome (cost): **XGB**

Conclusion: selecting by RMSE alone is suboptimal for operations.

---

# Answer to Dissertation Question
Yes, open transaction data can produce useful short-term demand forecasts.

And yes, those forecasts can improve inventory planning when connected to a cost-aware policy:
- lower weighted inventory cost
- controlled stockout risk
- measurable service-level behavior

---

# Practical Recommendation
Production candidate:
- Use **XGB** as decision model for ordering policy under asymmetric stockout risk.
- Keep **GBR** as forecasting benchmark.
- Tune reorder buffers per category from historical cost tradeoffs.

---

# Limitations and Next Steps
- Current final run used capped rows/category for Docker stability.
- Next step: full-scale rerun without caps and confidence intervals across rolling windows.
- Add business KPI simulation: fill rate, waste, and replenishment frequency.

---

# Reproducibility
Run from project root:

1. docker compose build
2. docker compose run --rm --remove-orphans trainer sh -lc "python src/run_final_stage.py"

Generated artifacts:
- src/models/baseline_benchmark_summary.csv
- src/models/inventory_impact_summary.csv
- docs/dissertation_results_final.md

---

# Thank You
Questions?
