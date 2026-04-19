# Stockout-Aware Retail Demand Forecasting

This project focuses on demand forecasting for perishable retail products using the FreshRetailNet-50K dataset. The pipeline is designed around stockout-aware preprocessing so forecasts better reflect latent demand, not just observed sales.

## What This Repository Contains

- Data preparation from raw hourly records to model-ready daily datasets
- Stockout-aware imputation and feature engineering
- Modular model pipelines for benchmark experiments
- Evaluation artifacts for forecasting quality and inventory impact

## Dataset

- Source: FreshRetailNet-50K
- Link: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- Domain: multi-store, perishable retail demand

## Project Layout

```text
.
├── src/
│   ├── ingest_flatten.py
│   ├── aggregate_impute.py
│   ├── featurize.py
│   ├── train_baseline_benchmarks.py
│   ├── run_final_stage.py
│   ├── evaluate_inventory_impact.py
│   └── model_pipelines/
├── notebooks/
├── docs/
├── requirements.txt
└── pyproject.toml
```

## Outputs

Running the project writes artifacts under `src/models/`, including benchmark predictions, per-category metrics, and publication-style reports.

## Commands and Usage

For all command-line usage, model-specific training/prediction commands, and benchmark orchestration, see:

- [src/model_pipelines/README.md](src/model_pipelines/README.md)
