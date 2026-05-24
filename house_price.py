"""
====================================================
  House Price Prediction — ML Project
  By: Ahmed Adel Shosha
====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   HOUSE PRICE PREDICTION — ML PROJECT")
print("   By: Ahmed Adel Shosha")
print("=" * 55)

# =====================================================
# 1. CREATE DATASET
# =====================================================
print("\n[1] Generating dataset...")

np.random.seed(42)
n = 1000

data = {
    'area_sqft':       np.random.randint(500,  5000, n),
    'bedrooms':        np.random.randint(1,    6,    n),
    'bathrooms':       np.random.randint(1,    4,    n),
    'age_years':       np.random.randint(0,    50,   n),
    'distance_center': np.round(np.random.uniform(0.5, 30, n), 1),
    'has_garage':      np.random.randint(0,    2,    n),
    'has_pool':        np.random.randint(0,    2,    n),
    'floor_number':    np.random.randint(1,    20,   n),
    'neighborhood':    np.random.choice(['Premium','Standard','Budget'], n, p=[0.3,0.5,0.2]),
}

df = pd.DataFrame(data)

# Realistic price formula
neighborhood_bonus = df['neighborhood'].map({'Premium': 80000, 'Standard': 20000, 'Budget': -30000})
df['price'] = (
    df['area_sqft']       * 120 +
    df['bedrooms']        * 15000 +
    df['bathrooms']       * 10000 -
    df['age_years']       * 1500 -
    df['distance_center'] * 3000 +
    df['has_garage']      * 25000 +
    df['has_pool']        * 35000 +
    df['floor_number']    * 2000 +
    neighborhood_bonus +
    np.random.normal(0, 15000, n)
).astype(int)

# Keep positive prices only
df = df[df['price'] > 50000].reset_index(drop=True)

print(f"   Dataset created: {len(df)} houses")
print(f"   Price range: ${df['price'].min():,} — ${df['price'].max():,}")
print(f"   Average price: ${df['price'].mean():,.0f}")

# =====================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
print("\n[2] Exploratory Data Analysis...")
print(df.describe().round(2))

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
print("\n[3] Feature Engineering...")

# Encode neighborhood
df['neighborhood_enc'] = df['neighborhood'].map({'Premium': 2, 'Standard': 1, 'Budget': 0})
df.drop('neighborhood', axis=1, inplace=True)

# New features
df['price_per_sqft'] = (df['price'] / df['area_sqft']).round(2)
df['rooms_total']    = df['bedrooms'] + df['bathrooms']
df['is_new']         = (df['age_years'] < 5).astype(int)

print("   New features added: price_per_sqft, rooms_total, is_new")

# =====================================================
# 4. PREPARE DATA
# =====================================================
print("\n[4] Preparing data...")

features = [
    'area_sqft', 'bedrooms', 'bathrooms', 'age_years',
    'distance_center', 'has_garage', 'has_pool',
    'floor_number', 'neighborhood_enc', 'rooms_total', 'is_new'
]

X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Training set : {len(X_train)} houses")
print(f"   Testing set  : {len(X_test)} houses")

# =====================================================
# 5. TRAIN MODELS
# =====================================================
print("\n[5] Training models...")

models = {
    'Linear Regression':     LinearRegression(),
    'Random Forest':         RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':     GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse= np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {'model': model, 'y_pred': y_pred, 'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"\n   {name}:")
    print(f"     R² Score : {r2:.4f}  ({r2*100:.1f}% accuracy)")
    print(f"     MAE      : ${mae:,.0f}")
    print(f"     RMSE     : ${rmse:,.0f}")

# =====================================================
# 6. BEST MODEL
# =====================================================
best_name = max(results, key=lambda k: results[k]['R2'])
best      = results[best_name]

print(f"\n{'='*55}")
print(f"  BEST MODEL: {best_name}")
print(f"  R² Score  : {best['R2']*100:.1f}%")
print(f"  MAE       : ${best['MAE']:,.0f}")
print(f"{'='*55}")

# =====================================================
# 7. FEATURE IMPORTANCE
# =====================================================
if best_name == 'Random Forest':
    importances = best['model'].feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False)
    print("\n[6] Feature Importances:")
    for _, row in feat_df.iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"   {row['Feature']:20s} {bar} {row['Importance']:.3f}")

# =====================================================
# 8. VISUALIZATIONS
# =====================================================
print("\n[7] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0a')
for ax in axes.flat:
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#888')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['right'].set_color('#333')

ACCENT = '#e8ff47'
ACCENT2 = '#ff6b35'

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test, best['y_pred'], alpha=0.4, color=ACCENT, s=15)
mn, mx = y_test.min(), y_test.max()
ax1.plot([mn, mx], [mn, mx], color=ACCENT2, linewidth=2, linestyle='--')
ax1.set_title('Actual vs Predicted Price', color='white', fontsize=13, pad=12)
ax1.set_xlabel('Actual Price ($)', color='#888')
ax1.set_ylabel('Predicted Price ($)', color='#888')

# Plot 2: Price Distribution
ax2 = axes[0, 1]
ax2.hist(df['price'], bins=40, color=ACCENT, alpha=0.8, edgecolor='#0a0a0a')
ax2.set_title('Price Distribution', color='white', fontsize=13, pad=12)
ax2.set_xlabel('Price ($)', color='#888')
ax2.set_ylabel('Count', color='#888')

# Plot 3: Model Comparison
ax3 = axes[1, 0]
model_names = list(results.keys())
r2_scores   = [results[m]['R2'] * 100 for m in model_names]
short_names = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
colors = [ACCENT if m == best_name else '#333' for m in model_names]
bars = ax3.bar(short_names, r2_scores, color=colors, edgecolor='#222', linewidth=0.5)
ax3.set_title('Model Comparison (R² Score %)', color='white', fontsize=13, pad=12)
ax3.set_ylabel('R² Score (%)', color='#888')
ax3.set_ylim(0, 105)
for bar, score in zip(bars, r2_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{score:.1f}%', ha='center', va='bottom', color='white', fontsize=10)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
if best_name in ['Random Forest', 'Gradient Boosting']:
    importances = best['model'].feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance').tail(8)
    ax4.barh(feat_df['Feature'], feat_df['Importance'], color=ACCENT, alpha=0.85)
    ax4.set_title('Feature Importance', color='white', fontsize=13, pad=12)
    ax4.set_xlabel('Importance', color='#888')
else:
    ax4.text(0.5, 0.5, 'N/A for Linear Regression',
             ha='center', va='center', color='#888', fontsize=12)

plt.suptitle('House Price Prediction — ML Project\nBy: Ahmed Adel Shosha',
             color='white', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/house_price_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("   Saved: house_price_results.png")

# =====================================================
# 9. PREDICT NEW HOUSE
# =====================================================
print("\n[8] Predicting a new house...")

new_house = pd.DataFrame([{
    'area_sqft':        1800,
    'bedrooms':         3,
    'bathrooms':        2,
    'age_years':        5,
    'distance_center':  8.0,
    'has_garage':       1,
    'has_pool':         0,
    'floor_number':     3,
    'neighborhood_enc': 1,   # Standard
    'rooms_total':      5,
    'is_new':           0,
}])

if best_name == 'Linear Regression':
    new_scaled   = scaler.transform(new_house)
    predicted    = best['model'].predict(new_scaled)[0]
else:
    predicted    = best['model'].predict(new_house)[0]

print(f"\n   House Details:")
print(f"     Area        : 1800 sqft")
print(f"     Bedrooms    : 3")
print(f"     Bathrooms   : 2")
print(f"     Age         : 5 years")
print(f"     Distance    : 8 km from center")
print(f"     Garage      : Yes")
print(f"     Neighborhood: Standard")
print(f"\n   Predicted Price: ${predicted:,.0f}")

print(f"\n{'='*55}")
print("  PROJECT COMPLETE!")
print(f"  Model: {best_name} | Accuracy: {best['R2']*100:.1f}%")
print("=" * 55)
