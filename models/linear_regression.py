import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_loader import df
warnings.filterwarnings('ignore')

print(df.head())

# Convert datetime

df['Date'] = pd.to_datetime(df['Date'],format="%d-%m-%Y")
df['Week']  = df['Date'].dt.isocalendar().week.astype(int)
df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year
df.drop(columns=['Date'], inplace=True)

le = LabelEncoder()
df['Holiday_Flag'] = le.fit_transform(df['Holiday_Flag'])

target      = 'Weekly_Sales'
continuous  = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
ordinal     = ['Week', 'Month']
binary      = ['Holiday_Flag']
categorical = ['Store']

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PEARSON — Continuous variables
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("PEARSON CORRELATION — Continuous Variables")
print("=" * 55)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Weekly Sales vs Continuous Variables (Pearson)', fontsize=13, fontweight='bold')

for ax, col in zip(axes.flat, continuous):
    r, p = stats.pearsonr(df[col], df[target])
    print(f"  {col:<20} r = {r:+.4f}  p = {p:.4f}  {'*' if p < 0.05 else ''}")
    sns.regplot(data=df, x=col, y=target, ax=ax,
                scatter_kws={'alpha': 0.3, 's': 10},
                line_kws={'color': 'crimson', 'linewidth': 2})
    ax.set_title(f'{col}  |  r = {r:+.3f}, p = {p:.4f}', fontsize=10)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. KENDALL — Ordinal variables
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("KENDALL'S TAU — Ordinal Variables")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Weekly Sales vs Ordinal Variables (Kendall)', fontsize=13, fontweight='bold')

for ax, col in zip(axes, ordinal):
    tau, p = stats.kendalltau(df[col], df[target])
    print(f"  {col:<20} τ = {tau:+.4f}  p = {p:.4f}  {'*' if p < 0.05 else ''}")
    mean_sales = df.groupby(col)[target].mean().reset_index()
    ax.scatter(df[col], df[target], alpha=0.2, s=8, color='steelblue')
    ax.plot(mean_sales[col], mean_sales[target], color='crimson', linewidth=2, marker='o', markersize=4)
    ax.set_title(f'{col}  |  τ = {tau:+.3f}, p = {p:.4f}', fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel('Weekly Sales')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TWO-SAMPLE T-TEST — Holiday_Flag
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("TWO-SAMPLE T-TEST — Holiday_Flag")
print("=" * 55)

group0 = df[df['Holiday_Flag'] == 0][target]
group1 = df[df['Holiday_Flag'] == 1][target]

# Levene's test for equal variances
lev_stat, lev_p = stats.levene(group0, group1)
equal_var = lev_p > 0.05
t_stat, p_ttest = stats.ttest_ind(group0, group1, equal_var=equal_var)

print(f"  Levene's test: p = {lev_p:.4f} → {'equal variances' if equal_var else 'unequal variances (Welch)'}")
print(f"  t = {t_stat:.4f},  p = {p_ttest:.4f}  {'*' if p_ttest < 0.05 else ''}")
print(f"  Mean non-holiday: ${group0.mean():,.0f}  |  Mean holiday: ${group1.mean():,.0f}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='Holiday_Flag', y=target,
            palette=['#5B9BD5', '#E15759'], ax=ax)
ax.set_xticklabels(['Non-Holiday (0)', 'Holiday (1)'])
ax.set_title(f'Weekly Sales by Holiday Flag\nt = {t_stat:.3f}, p = {p_ttest:.4f}', fontweight='bold')
ax.set_ylabel('Weekly Sales')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANOVA + KRUSKAL-WALLIS — Store
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ANOVA + KRUSKAL-WALLIS — Store")
print("=" * 55)

groups  = [g[target].values for _, g in df.groupby('Store')]
f_stat, p_anova = stats.f_oneway(*groups)

# Effect size η²
grand_mean = df[target].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in [pd.Series(g) for g in groups])
ss_total   = sum((df[target] - grand_mean) ** 2)
eta_sq     = ss_between / ss_total

print(f"  ANOVA:          F = {f_stat:.4f},  p = {p_anova:.4f}  {'*' if p_anova < 0.05 else ''}")
print(f"  Effect size η²= {eta_sq:.4f} ({'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'})")

# Normality check on residuals → decide whether to report Kruskal-Wallis
residuals = df[target] - df.groupby('Store')[target].transform('mean')
_, p_shapiro = stats.shapiro(residuals.sample(500, random_state=42))
print(f"  Shapiro-Wilk on residuals: p = {p_shapiro:.4f}")

if p_shapiro < 0.05:
    h_stat, p_kruskal = stats.kruskal(*groups)
    print(f"  → Non-normal residuals: reporting Kruskal-Wallis")
    print(f"  Kruskal-Wallis: H = {h_stat:.4f},  p = {p_kruskal:.4f}  {'*' if p_kruskal < 0.05 else ''}")

store_order = df.groupby('Store')[target].median().sort_values().index
fig, ax = plt.subplots(figsize=(20, 5))
sns.boxplot(data=df, x='Store', y=target, order=store_order,
            palette='Blues', ax=ax)
ax.set_title(f'Weekly Sales by Store (sorted by median)\n'
             f'ANOVA F = {f_stat:.2f}, p = {p_anova:.4f}, η² = {eta_sq:.3f}',
             fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Variable':<20} {'Test':<22} {'Statistic':>12}  {'p-value':>10}  Sig.")
print("-" * 55)
for col in continuous:
    r, p = stats.pearsonr(df[col], df[target])
    print(f"{col:<20} {'Pearson r':<22} {r:>+12.4f}  {p:>10.4f}  {'*' if p < 0.05 else ''}")
for col in ordinal:
    tau, p = stats.kendalltau(df[col], df[target])
    print(f"{col:<20} {'Kendall tau':<22} {tau:>+12.4f}  {p:>10.4f}  {'*' if p < 0.05 else ''}")
print(f"{'Holiday_Flag':<20} {'Two-sample t-test':<22} {t_stat:>+12.4f}  {p_ttest:>10.4f}  {'*' if p_ttest < 0.05 else ''}")
print(f"{'Store':<20} {'ANOVA F':<22} {f_stat:>+12.4f}  {p_anova:>10.4f}  {'*' if p_anova < 0.05 else ''}")

# ═══════════════════════════════════════════════════════════════════════════════
# Drop non-associated columns and split into train/test
# ═══════════════════════════════════════════════════════════════════════════════
# Significance threshold — keep only variables with p < 0.05
sig_cols = []

for col in continuous + ordinal:
    if col in continuous:
        _, p = stats.pearsonr(df[col], df[target])
    else:
        _, p = stats.kendalltau(df[col], df[target])
    if p < 0.05:
        sig_cols.append(col)
    print(f"  {col:<20} p = {p:.4f}  {'→ KEEP' if p < 0.05 else '→ DROP'}")

# Holiday_Flag (t-test)
if p_ttest < 0.05:
    sig_cols.append('Holiday_Flag')
print(f"  {'Holiday_Flag':<20} p = {p_ttest:.4f}  {'→ KEEP' if p_ttest < 0.05 else '→ DROP'}")

# Store (ANOVA)
if p_anova < 0.05:
    sig_cols.append('Store')
print(f"  {'Store':<20} p = {p_anova:.4f}  {'→ KEEP' if p_anova < 0.05 else '→ DROP'}")

print(f"\nRetained features: {sig_cols}")

# ── Build modelling dataframe ─────────────────────────────────────────────────
df_model = df[sig_cols + [target]].copy()

# One-hot encode Store if retained (drop_first avoids multicollinearity)
if 'Store' in sig_cols:
    df_model = pd.get_dummies(df_model, columns=['Store'], drop_first=True)

print(f"Shape after encoding: {df_model.shape}")

# ── Train / test split (80 / 20, stratify not needed for regression) ──────────
X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nX_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}  |  y_test: {y_test.shape}")