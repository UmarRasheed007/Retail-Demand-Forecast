# Publication Benchmark Report

## Project Title
Data-Driven Retail Demand Forecasting Using Machine Learning for Inventory Optimisation

## Research Question
How accurately can machine learning and time-series forecasting models predict short-term retail product demand using open transaction data, and how can these predictions be used to improve inventory planning?

## Experimental Setup
- Categories evaluated: 81, 60, 82, 184, 1
- Forecast horizon: 7 days
- Models benchmarked: lgbm, xgb, catboost, rf, extra_trees, gbr, ensemble_mean

## Forecast Accuracy Leaderboard
| Rank | Model | RMSE | MAE | MAPE | categories_evaluated |
|---|---|---|---|---|---|
| 1 | lgbm | 2.0507 | 1.2789 | 50.57% | 5 |
| 2 | xgb | 2.0911 | 1.3069 | 54.07% | 5 |
| 3 | catboost | 2.2368 | 1.3865 | 55.14% | 5 |
| 4 | rf | 2.2799 | 1.3673 | 50.52% | 5 |
| 5 | extra_trees | 2.2918 | 1.3938 | 53.93% | 5 |
| 6 | gbr | 2.4132 | 1.5321 | 63.45% | 5 |
| 7 | ensemble_mean | 5.9331 | 5.3641 | 53.32% | 5 |

## Inventory Scenarios
| Scenario | Model | Weighted Cost | Service Level | Stockout Rate | categories_evaluated |
|---|---|---|---|---|---|
| under=5, over=1 | ensemble_mean | 129.5611 | 0.629 | 0.371 | 5 |
| under=5, over=1 | lgbm | 16309.2739 | 0.723 | 0.277 | 5 |
| under=5, over=1 | xgb | 16920.3899 | 0.728 | 0.272 | 5 |
| under=5, over=1 | catboost | 17605.7754 | 0.744 | 0.256 | 5 |
| under=5, over=1 | rf | 17753.0998 | 0.745 | 0.255 | 5 |
| under=5, over=1 | extra_trees | 18320.2644 | 0.727 | 0.273 | 5 |
| under=5, over=1 | gbr | 19425.1613 | 0.747 | 0.253 | 5 |
| under=3, over=1 | ensemble_mean | 80.5564 | 0.629 | 0.371 | 5 |
| under=3, over=1 | lgbm | 12150.9138 | 0.692 | 0.308 | 5 |
| under=3, over=1 | xgb | 12669.1131 | 0.701 | 0.299 | 5 |
| under=3, over=1 | catboost | 13161.1406 | 0.704 | 0.296 | 5 |
| under=3, over=1 | rf | 13322.5011 | 0.709 | 0.291 | 5 |
| under=3, over=1 | extra_trees | 13520.6198 | 0.691 | 0.309 | 5 |
| under=3, over=1 | gbr | 14516.7291 | 0.732 | 0.268 | 5 |
| under=1, over=1 | ensemble_mean | 31.0584 | 0.629 | 0.371 | 5 |
| under=1, over=1 | lgbm | 6978.6792 | 0.555 | 0.445 | 5 |
| under=1, over=1 | xgb | 7256.0898 | 0.565 | 0.435 | 5 |
| under=1, over=1 | catboost | 7534.4758 | 0.582 | 0.418 | 5 |
| under=1, over=1 | rf | 7541.6436 | 0.558 | 0.442 | 5 |
| under=1, over=1 | extra_trees | 7609.3626 | 0.567 | 0.433 | 5 |
| under=1, over=1 | gbr | 8286.4906 | 0.587 | 0.413 | 5 |

## Key Finding
Best forecasting model by RMSE: lgbm (RMSE=2.0507, MAE=1.2789, MAPE=50.57%)

## Artifact Location
src/models/publication_benchmark
