# Model Pipelines Layout

This folder is the modular entrypoint for model training, prediction, benchmarking, and publication outputs.

## Structure

- baselines/
  - core.py
- lgbm/, rf/, extra_trees/, gbr/, xgb/, catboost/
  - train.py
  - predict.py
  - pipeline.py
- utils/
  - data.py
  - metrics.py
  - plotting.py
  - env.py
- benchmark_analysis.py (backward-compatible wrapper)
- benchmark_analysis_publication.py (full publication-grade benchmark)

## Naming Convention for Figures

Per-model forecast figure names are explicit:

- forecast_lgbm_category_grid.png
- forecast_rf_category_grid.png
- forecast_extra_trees_category_grid.png
- forecast_gbr_category_grid.png
- forecast_xgb_category_grid.png
- forecast_catboost_category_grid.png

Global comparison figures:

- comparison_rmse_all_models.png
- comparison_mae_all_models.png
- comparison_mape_all_models.png
- diagnostic_residual_histograms_all_models.png

## Recommended Output Directory

- src/models/publication_benchmark/
  - full_benchmark_predictions.csv
  - full_benchmark_per_category_metrics.csv
  - full_benchmark_summary.csv
  - publication_benchmark_report.md
  - figures/
  - inventory_scenarios/

## CLI Commands (Python equivalent of package.json scripts)

Command aliases are defined in [pyproject.toml](../../pyproject.toml) under `[project.scripts]`.

One-time setup from repo root:

```powershell
pip install -e .
```

After this, you can run commands directly:

```powershell
# LGBM
trainlgbm --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/lgbm
predictlgbm --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_eval_modelready.parquet --daily-path src/data/daily_dataset/daily_df_eval.parquet --flat-dir src/data/flattened_chunks_eval --test-days 10 --output-dir src/models/benchmark_artifacts/lgbm
pipelinelgbm --cats 81 60 82 184 1 --split train --batch-size 12000 --flat-dir src/data/flattened_chunks --daily-path src/data/daily_dataset/daily_df_imputed.parquet --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/lgbm

# RF
trainrf --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/rf
predictrf --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_eval_modelready.parquet --daily-path src/data/daily_dataset/daily_df_eval.parquet --flat-dir src/data/flattened_chunks_eval --test-days 10 --output-dir src/models/benchmark_artifacts/rf
pipelinerf --cats 81 60 82 184 1 --split train --batch-size 12000 --flat-dir src/data/flattened_chunks --daily-path src/data/daily_dataset/daily_df_imputed.parquet --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/rf

# Extra Trees
trainextratrees --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/extra_trees
predictextratrees --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_eval_modelready.parquet --daily-path src/data/daily_dataset/daily_df_eval.parquet --flat-dir src/data/flattened_chunks_eval --test-days 10 --output-dir src/models/benchmark_artifacts/extra_trees
pipelineextratrees --cats 81 60 82 184 1 --split train --batch-size 12000 --flat-dir src/data/flattened_chunks --daily-path src/data/daily_dataset/daily_df_imputed.parquet --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/extra_trees

# GBR
traingbr --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/gbr
predictgbr --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_eval_modelready.parquet --daily-path src/data/daily_dataset/daily_df_eval.parquet --flat-dir src/data/flattened_chunks_eval --test-days 10 --output-dir src/models/benchmark_artifacts/gbr
pipelinegbr --cats 81 60 82 184 1 --split train --batch-size 12000 --flat-dir src/data/flattened_chunks --daily-path src/data/daily_dataset/daily_df_imputed.parquet --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/gbr

# XGB
trainxgb --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/xgb
predictxgb --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_eval_modelready.parquet --daily-path src/data/daily_dataset/daily_df_eval.parquet --flat-dir src/data/flattened_chunks_eval --test-days 10 --output-dir src/models/benchmark_artifacts/xgb
pipelinexgb --cats 81 60 82 184 1 --split train --batch-size 12000 --flat-dir src/data/flattened_chunks --daily-path src/data/daily_dataset/daily_df_imputed.parquet --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/xgb

# CatBoost
traincatboost --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/catboost
predictcatboost --cats 81 60 82 184 1 --modelready-path src/data/daily_dataset/daily_df_eval_modelready.parquet --daily-path src/data/daily_dataset/daily_df_eval.parquet --flat-dir src/data/flattened_chunks_eval --test-days 10 --output-dir src/models/benchmark_artifacts/catboost
pipelinecatboost --cats 81 60 82 184 1 --split train --batch-size 12000 --flat-dir src/data/flattened_chunks --daily-path src/data/daily_dataset/daily_df_imputed.parquet --modelready-path src/data/daily_dataset/daily_df_modelready.parquet --test-days 10 --output-dir src/models/benchmark_artifacts/catboost

# Benchmarks
benchmark --cats 81 60 82 184 1 --test-days 7
```
