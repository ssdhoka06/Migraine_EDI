# debug_data_leakage.py
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

print("="*80)
print("🔍 DATA LEAKAGE DETECTION")
print("="*80)

# Load original data
df = pd.read_csv("final_training_dataset.csv")
print(f"\n📊 Original dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)[:20]}...")  # Show first 20

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Get feature names from preprocessor
preprocessor = joblib.load("foundation_preprocessor.joblib")

# Check what features are being used
print("\n🔍 Checking for suspicious features...\n")

# Suspicious keywords
leakage_keywords = [
    'diagnosis', 'migraine', 'midas', 'hit-6', 'hit6', 
    'treatment', 'medication', 'prescribed', 'doctor',
    'clinic', 'hospital', 'severity', 'disability'
]

print("❌ POTENTIAL LEAKAGE FEATURES:")
suspicious = []
for col in df.columns:
    col_lower = col.lower()
    for keyword in leakage_keywords:
        if keyword in col_lower:
            suspicious.append(col)
            print(f"   • {col}")
            break

# Feature importance check
print("\n🎯 Training quick model to check feature importance...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_names = []
for name, transformer, cols in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(cols)
    elif name == 'cat':
        feature_names.extend([f"{col}_{i}" for col in cols for i in range(10)])  # Approximate

# Top features
if hasattr(rf, 'feature_importances_'):
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    
    print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
    for i, idx in enumerate(top_idx, 1):
        if idx < len(feature_names):
            print(f"   {i}. Feature {idx}: importance={importances[idx]:.4f}")
        else:
            print(f"   {i}. Feature {idx}: importance={importances[idx]:.4f}")

# Check if any single feature predicts perfectly
print("\n⚡ Checking for perfect predictors...")
from sklearn.metrics import roc_auc_score

for i in range(min(50, X_train.shape[1])):  # Check first 50 features
    try:
        auc = roc_auc_score(y_train, X_train[:, i])
        if auc > 0.95 or auc < 0.05:
            print(f"   🚨 Feature {i}: AUC={auc:.4f} (PERFECT PREDICTOR!)")
    except:
        pass

print("\n" + "="*80)