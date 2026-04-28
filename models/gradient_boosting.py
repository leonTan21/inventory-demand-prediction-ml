import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import df

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
df['Date']  = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Week']  = df['Date'].dt.isocalendar().week.astype(int)
df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year
df.drop(columns=['Date'], inplace=True)

le = LabelEncoder()
df['Holiday_Flag'] = le.fit_transform(df['Holiday_Flag'])

print("Dtypes after feature engineering:")
print(df.dtypes)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TARGET LOG-TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
target = 'Weekly_Sales'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df[target], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Weekly_Sales (original scale)', fontsize=12)
axes[0].set_xlabel('Weekly Sales ($)')
axes[0].set_ylabel('Frequency')

log_sales = np.log(df[target])
axes[1].hist(log_sales, bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('log(Weekly_Sales)', fontsize=12)
axes[1].set_xlabel('log(Weekly Sales)')
axes[1].set_ylabel('Frequency')

plt.suptitle('Target Variable Distribution: Before and After Log-Transform', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Original skewness: {df[target].skew():.4f}")
print(f"Log-transformed skewness: {log_sales.skew():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION & ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
# Drop Fuel_Price (Pearson p = 0.4478 from statistical analysis — not significant)
df.drop(columns=['Fuel_Price'], inplace=True)

# One-hot encode Store (nominal variable with 45 categories)
df_model = pd.get_dummies(df, columns=['Store'], drop_first=True)

X = df_model.drop(columns=[target])
y = np.log(df_model[target])   # log-transform target

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()[:10]} ... (total {len(X.columns)})")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'X_train: {X_train.shape}  |  X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}  |  y_test: {y_test.shape}')

# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING (GridSearchCV)
# ═══════════════════════════════════════════════════════════════════════════════
# Gradient Boosting does not require feature scaling — tree-based models are scale-invariant.
gb = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample'    : [0.8, 1.0]
}

gb_grid = GridSearchCV(
    estimator  = gb,
    param_grid = param_grid,
    scoring    = 'r2',
    cv         = 5,
    n_jobs     = -1,
    verbose    = 1
)

gb_grid.fit(X_train, y_train)

print("\nBest parameters:")
for k, v in gb_grid.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nBest CV R²: {gb_grid.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
best_gb = gb_grid.best_estimator_

y_pred_train_log = best_gb.predict(X_train)
y_pred_test_log  = best_gb.predict(X_test)

# Metrics on log scale (comparable to Kaggle benchmark)
train_r2_log  = r2_score(y_train, y_pred_train_log)
test_r2_log   = r2_score(y_test,  y_pred_test_log)
test_rmse_log = mean_squared_error(y_test, y_pred_test_log) ** 0.5

# Convert back to original scale for interpretability
y_pred_test_orig  = np.exp(y_pred_test_log)
y_test_orig       = np.exp(y_test)
y_pred_train_orig = np.exp(y_pred_train_log)
y_train_orig      = np.exp(y_train)

train_r2  = r2_score(y_train_orig, y_pred_train_orig)
test_r2   = r2_score(y_test_orig,  y_pred_test_orig)
test_mae  = mean_absolute_error(y_test_orig, y_pred_test_orig)
test_rmse = mean_squared_error(y_test_orig,  y_pred_test_orig) ** 0.5

print('=' * 50)
print('GRADIENT BOOSTING — Results (Log Scale)')
print('=' * 50)
print(f'  Train R²:         {train_r2_log:.4f}')
print(f'  Test  R²:         {test_r2_log:.4f}')
print(f'  Test  RMSE (log): {test_rmse_log:.4f}')
print()
print('=' * 50)
print('GRADIENT BOOSTING — Results (Original Scale)')
print('=' * 50)
print(f'  Train R²:   {train_r2:.4f}')
print(f'  Test  R²:   {test_r2:.4f}')
print(f'  Test  MAE:  ${test_mae:,.0f}')
print(f'  Test  RMSE: ${test_rmse:,.0f}')

# ═══════════════════════════════════════════════════════════════════════════════
# 7. OVERFITTING CHECK
# ═══════════════════════════════════════════════════════════════════════════════
gap = train_r2_log - test_r2_log
print(f"\nTrain R² (log): {train_r2_log:.4f}")
print(f"Test  R² (log): {test_r2_log:.4f}")
print(f"Gap:            {gap:.4f}")
if gap < 0.02:
    print("-> No significant overfitting.")
elif gap < 0.05:
    print("-> Slight overfitting. Consider reducing max_depth or n_estimators.")
else:
    print("-> Overfitting detected. Reduce model complexity.")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
importances = best_gb.feature_importances_
feat_df = pd.DataFrame({
    'Feature':    X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

top20 = feat_df.head(20)

plt.figure(figsize=(10, 7))
sns.barplot(data=top20, x='Importance', y='Feature', palette='Blues_r')
plt.title('Gradient Boosting — Top 20 Feature Importances', fontsize=13, fontweight='bold')
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.show()

print("\nTop 10 features:")
print(top20.head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 9. DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Predicted (log scale)
residuals = y_test.values - y_pred_test_log
axes[0].scatter(y_pred_test_log, residuals, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(0, color='crimson', linewidth=1.5)
axes[0].set_xlabel('Predicted (log scale)')
axes[0].set_ylabel('Residual')
axes[0].set_title('Residuals vs Predicted (log scale)')

# Actual vs Predicted (original scale)
axes[1].scatter(y_test_orig, y_pred_test_orig, alpha=0.3, s=10, color='darkorange')
lims = [min(y_test_orig.min(), y_pred_test_orig.min()),
        max(y_test_orig.max(), y_pred_test_orig.max())]
axes[1].plot(lims, lims, color='crimson', linewidth=1.5)
axes[1].set_xlabel('Actual Weekly Sales ($)')
axes[1].set_ylabel('Predicted Weekly Sales ($)')
axes[1].set_title(f'Actual vs Predicted  (R² = {test_r2:.3f})')

plt.suptitle('Gradient Boosting — Diagnostic Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════════════
train_sizes, train_scores, val_scores = learning_curve(
    best_gb, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring='r2', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training R²')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='steelblue')
plt.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation R²')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='darkorange')
plt.xlabel('Training Set Size')
plt.ylabel('R²')
plt.title('Learning Curve — Gradient Boosting', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Final training R²:   {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
print(f"Final validation R²: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. COMPARISON AGAINST LINEAR REGRESSION BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr_log  = lr.predict(X_test_sc)
y_pred_lr_orig = np.exp(y_pred_lr_log)

lr_r2_log   = r2_score(y_test, y_pred_lr_log)
lr_rmse_log = mean_squared_error(y_test, y_pred_lr_log) ** 0.5
lr_r2       = r2_score(y_test_orig, y_pred_lr_orig)
lr_mae      = mean_absolute_error(y_test_orig, y_pred_lr_orig)
lr_rmse     = mean_squared_error(y_test_orig, y_pred_lr_orig) ** 0.5

comparison = pd.DataFrame({
    'Model'         : ['Linear Regression (baseline)', 'Gradient Boosting'],
    'Test R² (log)' : [lr_r2_log,   test_r2_log],
    'RMSE (log)'    : [lr_rmse_log, test_rmse_log],
    'Test R²'       : [lr_r2,       test_r2],
    'Test MAE ($)'  : [f'${lr_mae:,.0f}',  f'${test_mae:,.0f}'],
    'Test RMSE ($)' : [f'${lr_rmse:,.0f}', f'${test_rmse:,.0f}']
})

print(comparison.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ['Linear Regression', 'Gradient Boosting']
colors = ['#4878CF', '#E8713A']

for ax, metric, vals in zip(
    axes,
    ['Test R² (log)', 'RMSE (log)'],
    [[lr_r2_log, test_r2_log], [lr_rmse_log, test_rmse_log]]
):
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.5)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Linear Regression vs Gradient Boosting', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
