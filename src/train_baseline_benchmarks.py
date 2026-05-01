#!/usr/bin/env python

import os
import argparse
import importlib
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _build_models(model_names: List[str], random_state: int = 42) -> Dict[str, object]:
    available_models: Dict[str, object] = {
        "rf": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        ),
        "gbr": GradientBoostingRegressor(random_state=random_state),
    }

    try:
        import lightgbm as lgb

        available_models["lgbm"] = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
    except Exception:
        pass

    try:
        xgb_module = importlib.import_module("xgboost")
        XGBRegressor = getattr(xgb_module, "XGBRegressor")
        available_models["xgb"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=1,
            objective="reg:squarederror",
            verbosity=0,
        )
    except Exception:
        pass

    try:
        catboost_module = importlib.import_module("catboost")
        CatBoostRegressor = getattr(catboost_module, "CatBoostRegressor")
        available_models["catboost"] = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_state=random_state,
            thread_count=1,
            verbose=False,
        )
    except Exception:
        pass

    missing = [name for name in model_names if name not in available_models]
    if missing:
        raise ValueError(
            f"Unknown or unavailable model(s): {missing}. "
            f"Supported: {sorted(list(available_models.keys()))}"
        )

    return {name: available_models[name] for name in model_names}


def _candidate_param_sets(model_name: str) -> List[dict]:
    # Small, high-impact grids to keep runtime reasonable while improving accuracy.
    if model_name == "rf":
        return [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2},
        ]
    if model_name == "extra_trees":
        return [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 14, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 20, "min_samples_leaf": 2},
        ]
    if model_name == "gbr":
        return [
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 1.0},
            {"n_estimators": 350, "learning_rate": 0.03, "max_depth": 2, "subsample": 0.9},
            {"n_estimators": 450, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8},
        ]
    if model_name == "lgbm":
        return [
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 1.0},
            {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.9},
            {"n_estimators": 700, "learning_rate": 0.02, "num_leaves": 63, "subsample": 0.8},
        ]
    if model_name == "xgb":
        return [
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 1.0},
            {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.9},
            {"n_estimators": 700, "learning_rate": 0.02, "max_depth": 4, "subsample": 0.8},
        ]
    if model_name == "catboost":
        return [
            {"iterations": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3},
            {"iterations": 500, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5},
            {"iterations": 700, "learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 7},
        ]
    return [{}]


def _split_by_last_unique_days(df_cat: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_days = np.sort(df_cat["dt"].dropna().unique())
    if unique_days.shape[0] < 2:
        return pd.DataFrame(), pd.DataFrame()

    effective_days = min(holdout_days, unique_days.shape[0] - 1)
    split_start_dt = pd.Timestamp(unique_days[-effective_days])
    train_df = df_cat[df_cat["dt"] < split_start_dt].copy()
    holdout_df = df_cat[df_cat["dt"] >= split_start_dt].copy()
    return train_df, holdout_df


def _fit_predict_rmse(model: object, x_tr: pd.DataFrame, y_tr: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> float:
    model.fit(x_tr, y_tr)
    pred = model.predict(x_val)
    return float(np.sqrt(mean_squared_error(y_val, pred)))


def _tune_model(
    model_name: str,
    base_model: object,
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> object:
    # Build a time-aware validation split from the training window only.
    unique_train_days = np.sort(train_df["dt"].dropna().unique())
    if unique_train_days.shape[0] < 8:
        return base_model

    val_days = max(2, min(7, int(unique_train_days.shape[0] * 0.2)))
    tr_sub, val_sub = _split_by_last_unique_days(train_df, holdout_days=val_days)
    if tr_sub.empty or val_sub.empty:
        return base_model

    x_tr = tr_sub[feature_cols]
    y_tr = tr_sub["daily_sale_imputed"]
    x_val = val_sub[feature_cols]
    y_val = val_sub["daily_sale_imputed"]

    best_rmse = float("inf")
    best_model = base_model

    for params in _candidate_param_sets(model_name):
        model = clone(base_model)
        if params:
            model.set_params(**params)
        try:
            score = _fit_predict_rmse(model, x_tr, y_tr, x_val, y_val)
        except Exception:
            continue
        if score < best_rmse:
            best_rmse = score
            best_model = model

    return best_model


def benchmark_models_for_categories(
    modelready_path: str,
    cats: List[int],
    model_names: List[str],
    output_dir: str,
    test_days: int = 10,
    max_rows_per_category: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(modelready_path):
        raise FileNotFoundError(f"No feature-ready parquet found at: {modelready_path}")

    df = pd.read_parquet(modelready_path)
    if "dt" not in df.columns or "daily_sale_imputed" not in df.columns:
        raise ValueError("Expected columns `dt` and `daily_sale_imputed` were not found.")

    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])

    models = _build_models(model_names=model_names, random_state=random_state)

    detailed_rows = []
    prediction_rows = []

    for cat in cats:
        df_cat = (
            df[df["third_category_id"] == cat]
            .sort_values("dt")
            .reset_index(drop=True)
        )

        if max_rows_per_category is not None and df_cat.shape[0] > max_rows_per_category:
            df_cat = df_cat.tail(max_rows_per_category).reset_index(drop=True)

        unique_days = np.sort(df_cat["dt"].dropna().unique())
        if unique_days.shape[0] < 2:
            print(
                f"⚠️ Skipping category {cat}: not enough unique dates "
                f"({unique_days.shape[0]})."
            )
            continue

        effective_test_days = min(test_days, unique_days.shape[0] - 1)
        if effective_test_days != test_days:
            print(
                f"⚠️ Category {cat}: requested test_days={test_days} but only "
                f"{unique_days.shape[0]} unique dates are available; using "
                f"test_days={effective_test_days}."
            )

        # Hold out the final N unique calendar days (not final N rows),
        # so panel-style data with many rows/day still gets a real horizon split.
        train_df, test_df = _split_by_last_unique_days(df_cat, holdout_days=effective_test_days)

        if train_df.empty or test_df.empty:
            print(
                f"⚠️ Skipping category {cat}: invalid split "
                f"(train_rows={train_df.shape[0]}, test_rows={test_df.shape[0]})."
            )
            continue

        feature_cols = [c for c in df_cat.columns if c not in ["daily_sale_imputed", "dt"]]
        x_train = train_df[feature_cols]
        y_train = train_df["daily_sale_imputed"]
        x_test = test_df[feature_cols]
        y_test = test_df["daily_sale_imputed"]

        for model_name, base_model in models.items():
            model = _tune_model(model_name, base_model, train_df, feature_cols)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            mape = _safe_mape(y_test.values, preds)

            detailed_rows.append(
                {
                    "category": cat,
                    "model": model_name,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                }
            )

            prediction_rows.extend(
                [
                    {
                        "category": cat,
                        "model": model_name,
                        "dt": dt,
                        "actual": float(actual),
                        "prediction": float(pred),
                        "error": float(actual - pred),
                        "abs_error": float(abs(actual - pred)),
                    }
                    for dt, actual, pred in zip(test_df["dt"].values, y_test.values, preds)
                ]
            )

    if not detailed_rows:
        raise RuntimeError("No benchmark results were produced. Check categories and data availability.")

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = (
        detailed_df.groupby("model", as_index=False)
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

    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, "baseline_benchmark_detailed.csv")
    summary_path = os.path.join(output_dir, "baseline_benchmark_summary.csv")
    predictions_path = os.path.join(output_dir, "baseline_benchmark_predictions.csv")
    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    best = summary_df.iloc[0]
    print("\n✅ Benchmark complete")
    print(f"Detailed results: {detailed_path}")
    print(f"Summary results:  {summary_path}")
    print(f"Predictions:      {predictions_path}")
    print(
        f"Best model by RMSE: {best['model']} "
        f"(RMSE={best['rmse_mean']:.4f}, MAE={best['mae_mean']:.4f})"
    )

    return detailed_df, summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple regressors on model-ready daily features."
    )
    parser.add_argument(
        "--modelready-path",
        default="data/daily_dataset/daily_df_modelready.parquet",
        help="Path to model-ready parquet.",
    )
    parser.add_argument(
        "--cats",
        nargs="+",
        type=int,
        required=True,
        help="third_category_id values to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "rf", "extra_trees", "gbr", "xgb", "catboost"],
        help="Model keys to benchmark. Example: lgbm rf extra_trees gbr xgb catboost",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=10,
        help="Number of final days per category used for testing.",
    )
    parser.add_argument(
        "--max-rows-per-category",
        type=int,
        default=None,
        help="Optional cap for rows per category (keeps most recent rows) to reduce memory/time.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to write benchmark CSV outputs.",
    )
    args = parser.parse_args()

    benchmark_models_for_categories(
        modelready_path=args.modelready_path,
        cats=args.cats,
        model_names=args.models,
        output_dir=args.output_dir,
        test_days=args.test_days,
        max_rows_per_category=args.max_rows_per_category,
    )


if __name__ == "__main__":
    main()