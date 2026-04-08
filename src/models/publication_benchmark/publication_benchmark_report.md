# Publication Benchmark Report

## Project Title
Data-Driven Retail Demand Forecasting Using Machine Learning for Inventory Optimisation

## Research Question
How accurately can machine learning and time-series forecasting models predict short-term retail product demand using open transaction data, and how can these predictions be used to improve inventory planning?

## Experimental Setup
- Categories evaluated: 81, 60, 82, 184, 1
- Forecast horizon: 7 days
- Models benchmarked: xgb, extra_trees, catboost, rf, gbr, lgbm, nbeats

## Forecast Accuracy Leaderboard
| Rank | Model | RMSE | MAE | MAPE | categories_evaluated |
|---|---|---|---|---|---|
| 1 | xgb | 1.0360 | 0.7332 | 40.39% | 5 |
| 2 | extra_trees | 1.2041 | 0.8830 | 42.31% | 5 |
| 3 | catboost | 1.2450 | 0.8510 | 42.96% | 5 |
| 4 | rf | 1.3166 | 0.9169 | 45.67% | 5 |
| 5 | gbr | 1.3394 | 1.0551 | 61.07% | 5 |
| 6 | lgbm | 1.3488 | 0.9365 | 48.31% | 5 |
| 7 | nbeats | 911.7447 | 712.7234 | 24.55% | 5 |

## Inventory Scenarios
| Scenario | Model | Weighted Cost | Service Level | Stockout Rate | categories_evaluated |
|---|---|---|---|---|---|
| under=5, over=1 | xgb | 8.5600 | 0.686 | 0.314 | 5 |
| under=5, over=1 | catboost | 9.6781 | 0.771 | 0.229 | 5 |
| under=5, over=1 | extra_trees | 10.3399 | 0.714 | 0.286 | 5 |
| under=5, over=1 | rf | 10.4346 | 0.686 | 0.314 | 5 |
| under=5, over=1 | lgbm | 11.6787 | 0.714 | 0.286 | 5 |
| under=5, over=1 | gbr | 11.8508 | 0.743 | 0.257 | 5 |
| under=5, over=1 | nbeats | 9203.2395 | 0.714 | 0.286 | 5 |
| under=3, over=1 | xgb | 7.2020 | 0.686 | 0.314 | 5 |
| under=3, over=1 | catboost | 8.1708 | 0.714 | 0.286 | 5 |
| under=3, over=1 | rf | 8.8290 | 0.657 | 0.343 | 5 |
| under=3, over=1 | extra_trees | 8.9666 | 0.629 | 0.371 | 5 |
| under=3, over=1 | lgbm | 9.4011 | 0.714 | 0.286 | 5 |
| under=3, over=1 | gbr | 9.4421 | 0.686 | 0.314 | 5 |
| under=3, over=1 | nbeats | 7254.2606 | 0.686 | 0.314 | 5 |
| under=1, over=1 | xgb | 5.0441 | 0.514 | 0.486 | 5 |
| under=1, over=1 | catboost | 5.8115 | 0.629 | 0.371 | 5 |
| under=1, over=1 | extra_trees | 6.1275 | 0.600 | 0.400 | 5 |
| under=1, over=1 | rf | 6.3643 | 0.600 | 0.400 | 5 |
| under=1, over=1 | lgbm | 6.4425 | 0.600 | 0.400 | 5 |
| under=1, over=1 | gbr | 6.7442 | 0.629 | 0.371 | 5 |
| under=1, over=1 | nbeats | 4298.0213 | 0.514 | 0.486 | 5 |

## Key Finding
Best forecasting model by RMSE: xgb (RMSE=1.0360, MAE=0.7332, MAPE=40.39%)

## Artifact Location
src/models/publication_benchmark
