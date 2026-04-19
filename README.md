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

## Generate src/data

`src/data` is not committed to GitHub. Generate it from the repository root with the steps below.

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
pip install -e .
```

3. Create flattened hourly parquet chunks for train split.

```powershell
python src/ingest_flatten.py --split train --output-dir src/data/flattened_chunks
```

4. Aggregate and impute daily train dataset.

```powershell
python src/aggregate_impute.py --input-dir src/data/flattened_chunks --output-path src/data/daily_dataset/daily_df_imputed.parquet
```

5. Build train model-ready features.

```powershell
python src/featurize.py --input-path src/data/daily_dataset/daily_df_imputed.parquet --output-path src/data/daily_dataset/daily_df_modelready.parquet
```

6. Create flattened hourly parquet chunks for eval split.

```powershell
python src/ingest_flatten.py --split eval --output-dir src/data/flattened_chunks_eval
```

7. Aggregate and impute daily eval dataset.

```powershell
python src/aggregate_impute.py --input-dir src/data/flattened_chunks_eval --output-path src/data/daily_dataset/daily_df_eval.parquet
```

8. Build eval model-ready features.

```powershell
python src/featurize.py --input-path src/data/daily_dataset/daily_df_eval.parquet --output-path src/data/daily_dataset/daily_df_eval_modelready.parquet
```

After these steps, you should have:

- `src/data/flattened_chunks/`
- `src/data/flattened_chunks_eval/`
- `src/data/daily_dataset/daily_df_imputed.parquet`
- `src/data/daily_dataset/daily_df_modelready.parquet`
- `src/data/daily_dataset/daily_df_eval.parquet`
- `src/data/daily_dataset/daily_df_eval_modelready.parquet`

## Commands and Usage

For all command-line usage, model-specific training/prediction commands, and benchmark orchestration, see:

- [src/model_pipelines/README.md](src/model_pipelines/README.md)
