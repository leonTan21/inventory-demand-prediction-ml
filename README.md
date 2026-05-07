# Walmart Weekly Sales — ML Project

This project predicts weekly sales across 45 Walmart stores using six machine learning regression models, comparing their performance on real retail data from Kaggle. This will help the stores with stock management.

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

## Results

All models share a fixed `random_state=42` and an 80/20 train/test split. The winner is **Random Forest** by Test R², with the lowest MAE and RMSE among all six models.

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.9272 | 0.9256 | $96,163 | $154,818 |
| Polynomial Regression (Degree 2) | 0.9563 | 0.9484 | $78,223 | $128,887 |
| Ridge Regression | 0.9272 | 0.9256 | $96,161 | $154,820 |
| XGBoost | 0.8409 | 0.8324 | $188,268 | $232,394 |
| **Random Forest** | **0.9942** | **0.9562** | **$64,214** | **$118,830** |
| Gradient Boosting | 0.9634 | 0.9446 | $85,187 | $133,577 |

Key takeaways:
- Random Forest achieves the best Test R² (0.9562) and lowest MAE ($64,214), explaining 95.6% of sales variance.
- Linear and Ridge Regression perform nearly identically — regularisation alone does not address the dataset's non-linearity.
- XGBoost underperforms with default settings; tuning `n_estimators`, `max_depth`, and `learning_rate` would improve it.

## Reproducing results

The preprocessing pipeline applied to every model:

- `Date` decomposed into `Week`, `Month`, `Year`
- `Fuel_Price` dropped (Pearson p = 0.4478, not significant)
- `Store` one-hot encoded (`drop_first=True`)
- `Weekly_Sales` log-transformed to reduce skewness; metrics reported on both log and original scale

| Model | File | Key settings |
|---|---|---|
| Linear Regression | `models/linear_regression.py` | Statistical feature selection (Pearson, Kendall, t-test, ANOVA); StandardScaler |
| Polynomial Regression | `models/polynomial_regression.py` | Degree-2 features; StandardScaler |
| Ridge Regression | `models/ridge.py` | GridSearchCV over alpha; StandardScaler |
| XGBoost | `models/xgboost_regression.py` | n_estimators=100, learning_rate=0.1, max_depth=3 |
| Random Forest | `models/random_forest.py` | n_estimators=200, max_features='sqrt'; OOB score |
| Gradient Boosting | `models/gradient_boosting.py` | n_estimators=200, max_depth=4, learning_rate=0.1 |

Running the same scripts on the same dataset with the settings above will reproduce the reported metrics exactly.
