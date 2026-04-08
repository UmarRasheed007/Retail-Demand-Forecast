#!/usr/bin/env python

import argparse
from train_baseline_benchmarks import benchmark_models_for_categories
from evaluate_inventory_impact import evaluate_inventory_impact


def main():
    parser = argparse.ArgumentParser(
        description="Run final-stage benchmark and inventory-impact evaluation."
    )
    parser.add_argument(
        "--modelready-path",
        default="src/data/daily_dataset/daily_df_modelready.parquet",
        help="Path to model-ready parquet.",
    )
    parser.add_argument(
        "--cats",
        nargs="+",
        type=int,
        default=[81, 60, 82, 184, 1],
        help="Categories to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "rf", "extra_trees", "gbr", "xgb", "catboost"],
        help="Models to benchmark.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=10,
        help="Last N rows per category used as test horizon.",
    )
    parser.add_argument(
        "--max-rows-per-category",
        type=int,
        default=2000,
        help="Cap rows per category to keep runs stable in constrained environments.",
    )
    parser.add_argument(
        "--understock-cost",
        type=float,
        default=5.0,
        help="Penalty weight for understock units.",
    )
    parser.add_argument(
        "--overstock-cost",
        type=float,
        default=1.0,
        help="Penalty weight for overstock units.",
    )
    parser.add_argument(
        "--output-dir",
        default="src/models",
        help="Directory to write outputs.",
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

    evaluate_inventory_impact(
        predictions_path=f"{args.output_dir}/baseline_benchmark_predictions.csv",
        output_dir=args.output_dir,
        understock_cost=args.understock_cost,
        overstock_cost=args.overstock_cost,
    )


if __name__ == "__main__":
    main()
