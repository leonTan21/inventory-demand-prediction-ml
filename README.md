# Walmart Weekly Sales — ML Project

This project predicts weekly sales across 45 Walmart stores using six machine learning regression models, comparing their performance on real retail data from Kaggle.

## Purpose

The goal is to forecast Walmart's weekly sales figures by applying and evaluating a range of regression techniques — from simple linear models to ensemble methods. Key functionalities include automated data loading via the Kaggle API, a shared preprocessing pipeline (date decomposition, one-hot encoding of stores, log-transformation of the target), statistical feature selection, hyperparameter tuning with GridSearchCV, and a side-by-side model comparison reported on both log and original sales scales.

## Project structure

```
.
├── data_loader.py                       # Loads Walmart.csv (cache or kagglehub)
├── models/
│   ├── linear_regression.py             # Linear Regression with statistical feature selection
│   ├── polynomial_regression.py         # Polynomial Regression (degree 2)
│   ├── ridge.py                         # Ridge Regression with GridSearchCV over alpha
│   ├── xgboost_regression.py            # XGBoost with GridSearchCV tuning
│   ├── random_forest.py                 # Random Forest with GridSearchCV tuning + OOB score
│   └── gradient_boosting.py             # Gradient Boosting with GridSearchCV tuning
├── notebooks/
│   └── walmart_sales_prediction.ipynb   # Full analysis: EDA, feature selection, all 6 models + comparison
├── Project_Report_Walmart_Sales.docx    # Written project report
└── Walmart_Sales_Prediction.pptx        # Presentation slides
```

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy kagglehub xgboost jupyter
   ```

3. **Configure Kaggle API access**

   A Kaggle API key is required for the first dataset download. Place `kaggle.json` in `~/.kaggle/`, or set the environment variables:

   ```bash
   set KAGGLE_USERNAME=your_username
   set KAGGLE_KEY=your_api_key
   ```

   Subsequent runs use the cached file at `~/.cache/kagglehub/`.

## Running the models

Run any individual model script from the project root:

```bash
python models/linear_regression.py
python models/polynomial_regression.py
python models/ridge.py
python models/xgboost_regression.py
python models/random_forest.py
python models/gradient_boosting.py
```

To run the full end-to-end analysis with EDA, feature selection, all six models, and a comparison table, open the notebook:

```bash
jupyter notebook notebooks/walmart_sales_prediction.ipynb
```

## Reproducing results

All experiments use a fixed `random_state=42` and an 80/20 train/test split. The preprocessing pipeline applied to every model is:

- `Date` decomposed into `Week`, `Month`, `Year`
- `Fuel_Price` dropped (Pearson p = 0.4478, not significant)
- `Store` one-hot encoded (`drop_first=True`)
- `Weekly_Sales` log-transformed to reduce skewness; metrics reported on both log and original scale

| Model | File | Key settings |
|---|---|---|
| Linear Regression | `models/linear_regression.py` | Statistical feature selection (Pearson, Kendall, t-test, ANOVA); StandardScaler |
| Polynomial Regression | `models/polynomial_regression.py` | Degree-2 features; StandardScaler |
| Ridge Regression | `models/ridge.py` | GridSearchCV over alpha; StandardScaler |
| XGBoost | `models/xgboost_regression.py` | GridSearchCV over n_estimators, max_depth, learning_rate, subsample |
| Random Forest | `models/random_forest.py` | GridSearchCV over n_estimators, max_features, max_depth, min_samples_split; OOB score |
| Gradient Boosting | `models/gradient_boosting.py` | GridSearchCV over n_estimators, max_depth, learning_rate, subsample |

Running the same scripts on the same dataset with the settings above will reproduce the reported metrics exactly.
