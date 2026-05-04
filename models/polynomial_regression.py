import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE SELECTION & ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
df.drop(columns=['Fuel_Price'], inplace=True)
df_model = pd.get_dummies(df, columns=['Store'], drop_first=True)

target = 'Weekly_Sales'
X = df_model.drop(columns=[target])
y = np.log(df_model[target])

print(f"Feature matrix shape: {X.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'X_train: {X_train.shape}  |  X_test: {X_test.shape}')

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN POLYNOMIAL REGRESSION (degree 2)
# ═══════════════════════════════════════════════════════════════════════════════
poly_model = Pipeline([
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

poly_model.fit(X_train, y_train)
print('Polynomial regression (degree 2) model trained.')

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
y_pred_train_log = poly_model.predict(X_train)
y_pred_test_log  = poly_model.predict(X_test)

train_r2_log  = r2_score(y_train, y_pred_train_log)
test_r2_log   = r2_score(y_test,  y_pred_test_log)
test_rmse_log = mean_squared_error(y_test, y_pred_test_log) ** 0.5

y_pred_test_orig  = np.exp(y_pred_test_log)
y_test_orig       = np.exp(y_test)
y_pred_train_orig = np.exp(y_pred_train_log)
y_train_orig      = np.exp(y_train)

train_r2  = r2_score(y_train_orig, y_pred_train_orig)
test_r2   = r2_score(y_test_orig,  y_pred_test_orig)
test_mae  = mean_absolute_error(y_test_orig, y_pred_test_orig)
test_rmse = mean_squared_error(y_test_orig,  y_pred_test_orig) ** 0.5

print('=' * 50)
print('POLYNOMIAL REGRESSION (deg 2) — Log Scale')
print('=' * 50)
print(f'  Train R²:         {train_r2_log:.4f}')
print(f'  Test  R²:         {test_r2_log:.4f}')
print(f'  Test  RMSE (log): {test_rmse_log:.4f}')
print()
print('=' * 50)
print('POLYNOMIAL REGRESSION (deg 2) — Original Scale')
print('=' * 50)
print(f'  Train R²:   {train_r2:.4f}')
print(f'  Test  R²:   {test_r2:.4f}')
print(f'  Test  MAE:  ${test_mae:,.0f}')
print(f'  Test  RMSE: ${test_rmse:,.0f}')

# ═══════════════════════════════════════════════════════════════════════════════
# 6. OVERFITTING CHECK
# ═══════════════════════════════════════════════════════════════════════════════
gap = train_r2_log - test_r2_log
print(f"\nTrain R² (log): {train_r2_log:.4f}")
print(f"Test  R² (log): {test_r2_log:.4f}")
print(f"Gap (train-test): {gap:.4f}")
if gap < 0.02:
    print("-> No significant overfitting.")
elif gap < 0.05:
    print("-> Slight overfitting.")
else:
    print("-> Overfitting detected. Consider reducing degree.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals = y_test.values - y_pred_test_log
axes[0].scatter(y_pred_test_log, residuals, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(0, color='crimson', linewidth=1.5)
axes[0].set_xlabel('Predicted (log scale)')
axes[0].set_ylabel('Residual')
axes[0].set_title('Residuals vs Predicted (log scale)')

axes[1].scatter(y_test_orig, y_pred_test_orig, alpha=0.3, s=10, color='steelblue')
lims = [min(y_test_orig.min(), y_pred_test_orig.min()),
        max(y_test_orig.max(), y_pred_test_orig.max())]
axes[1].plot(lims, lims, color='crimson', linewidth=1.5)
axes[1].set_xlabel('Actual Weekly Sales ($)')
axes[1].set_ylabel('Predicted Weekly Sales ($)')
axes[1].set_title(f'Actual vs Predicted  (R² = {test_r2:.3f})')

plt.suptitle('Polynomial Regression — Diagnostic Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
