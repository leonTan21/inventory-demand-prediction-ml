# Walmart Weekly Sales — ML Project

Predicts weekly sales across 45 Walmart stores using the [Walmart Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset) from Kaggle.

## Project structure

```
.
├── data_loader.py       # Loads Walmart.csv (cache or kagglehub)
├── models/              # One file per model
└── notebooks/           # Exploratory and model notebooks
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install pandas scikit-learn matplotlib seaborn scipy kagglehub
```

A Kaggle API key is required for the first download. Place `kaggle.json` in `~/.kaggle/` or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables. Subsequent runs use the cached file at `~/.cache/kagglehub/`.

## Models

| Model | File |
|---|---|
| Linear Regression | `models/linear_regression.py` |

## Usage

```bash
python models/<model>.py
```
