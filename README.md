# Walmart Weekly Sales — ML Project

Predicts weekly sales across 45 Walmart stores using the [Walmart Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset) from Kaggle.

## Project structure

```
.
├── data_loader.py                       # Loads Walmart.csv (cache or kagglehub)
├── models/
│   ├── linear_regression.py             # Linear Regression with statistical feature selection
│   ├── polynomial_regression.py         # Polynomial Regression (degree 2)
│   ├── ridge.py                         # Ridge Regression with GridSearchCV over alpha
│   ├── xgboost_regression.py            # XGBoost with GridSearchCV tuning
│   ├── random_forest.py                 # Random Forest with GridSearchCV tuning
│   └── gradient_boosting.py            # Gradient Boosting with GridSearchCV tuning
└── notebooks/
    └── walmart_linear_regression.ipynb  # Full analysis: EDA, feature selection, all 6 models + comparison
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn scipy kagglehub xgboost
```

A Kaggle API key is required for the first download. Place `kaggle.json` in `~/.kaggle/` or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables. Subsequent runs use the cached file at `~/.cache/kagglehub/`.

## Models

All models share the same preprocessing pipeline:
- `Date` decomposed into `Week`, `Month`, `Year`
- `Fuel_Price` dropped (Pearson p = 0.45, not significant)
- `Store` one-hot encoded (drop_first=True)
- Target `Weekly_Sales` log-transformed to reduce skewness; metrics reported on both log and original scale
- 80/20 train/test split with `random_state=42`

| Model | File | Notes |
|---|---|---|
| Linear Regression | `models/linear_regression.py` | Statistical feature selection (Pearson, Kendall, ANOVA); StandardScaler |
| Polynomial Regression | `models/polynomial_regression.py` | Degree-2 features; StandardScaler |
| Ridge Regression | `models/ridge.py` | GridSearchCV over alpha; StandardScaler |
| XGBoost | `models/xgboost_regression.py` | GridSearchCV over n_estimators, max_depth, learning_rate, subsample |
| Random Forest | `models/random_forest.py` | GridSearchCV over n_estimators, max_features, max_depth, min_samples_split; OOB score |
| Gradient Boosting | `models/gradient_boosting.py` | GridSearchCV over n_estimators, max_depth, learning_rate, subsample |

## Usage

```bash
python models/linear_regression.py
python models/polynomial_regression.py
python models/ridge.py
python models/xgboost_regression.py
python models/random_forest.py
python models/gradient_boosting.py

