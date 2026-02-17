# DengAI — Predicting Dengue Fever Outbreaks

My solution for [DrivenData's DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/), where the goal is to predict weekly dengue fever cases in two cities using climate data.

**Current rank: #1,016 out of 15,988 (top 6%)**

## The Problem

Given 20+ climate variables (temperature, humidity, precipitation, vegetation index, etc.) for San Juan, Puerto Rico and Iquitos, Peru — predict the number of dengue cases each week. Scored by Mean Absolute Error (MAE).

## Approach

The two cities have very different dengue patterns (San Juan peaks at 400+ cases, Iquitos rarely goes above 20), so I train separate models for each.

**Features I built:**
- Rolling averages and standard deviations of climate variables (4, 8, 12, 26 week windows)
- Cyclical encoding of week/month (sine and cosine transforms)
- Wet/dry season flags
- Temperature-humidity interaction terms
- Lagged climate features (1-4 weeks back — using only climate data, not target, to avoid leakage)

**Models:**
- Ensemble of XGBoost, LightGBM, and CatBoost
- XGBoost uses Poisson regression (count data objective) — makes sense since dengue cases are non-negative integers
- Time-series cross-validation (can't use random splits on sequential data)
- Predictions rounded to non-negative integers and clipped per city

## Results

| Version | MAE | Rank | What changed |
|---------|-----|------|-------------|
| v1 | 34.67 | 6,869 | Baseline — had target leakage from lag features |
| v2 | 24.54 | 1,016 | Fixed leakage, Poisson objective, deeper rolling features |

## What I Learned

- **Target leakage is sneaky.** My v1 used lagged target values as features, which works great on training data but produces zeros in the test set (you don't have future targets). Caught it by looking at feature values in the test predictions.
- **Count data needs count models.** Switching from squared error to Poisson regression made a real difference — dengue cases are counts, not continuous values.
- **City-specific models beat one-size-fits-all.** San Juan and Iquitos have completely different scales and seasonality. Training them together was hurting both.
- **Wide rolling windows help.** Dengue outbreaks respond to climate conditions from weeks or months ago, not just the current week. 12 and 26-week rolling features capture that delayed effect.

## How to Run

1. Open [the notebook in Google Colab](https://colab.research.google.com/github/ScottT2-spec/dengai-solution/blob/main/DengAI_Solution.ipynb)
2. Download the competition data from [DrivenData](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/)
3. Upload the 3 CSV files to your Google Drive
4. Update the `DATA_DIR` path in the notebook
5. Run all cells — generates `submission.csv`

## Files

- `DengAI_Solution.ipynb` — Colab notebook (run this)
- `dengai_solution.py` — Same code as a Python script

## Competition

- [DengAI on DrivenData](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
- Metric: Mean Absolute Error (MAE)
- 15,988 participants
