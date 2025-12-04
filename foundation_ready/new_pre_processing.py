# preprocessing_NO_LEAKAGE_FINAL.py
"""
Remove ALL leakage features based on comprehensive column analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

print("="*80)
print("🔄 LEAKAGE-FREE PREPROCESSING")
print("="*80)

# Load data
df = pd.read_csv("final_training_dataset.csv")
print(f"\nOriginal shape: {df.shape}")

# ============================================================================
# REMOVE ALL LEAKAGE FEATURES
# ============================================================================
print("\n🚫 Removing leakage features...")

# 1. All diagnosis columns
diagnosis_cols = [c for c in df.columns if 'Diagnosis' in c]
print(f"   Removing {len(diagnosis_cols)} diagnosis columns")

# 2. MIDAS columns
midas_cols = [c for c in df.columns if 'MIDAS' in c]
print(f"   Removing {len(midas_cols)} MIDAS columns")

# 3. HIT-6 columns  
hit6_cols = [c for c in df.columns if 'HIT6' in c or 'HIT 6' in c]
print(f"   Removing {len(hit6_cols)} HIT-6 columns")

# 4. GAD (anxiety) columns
gad_cols = [c for c in df.columns if 'GAD' in c]
print(f"   Removing {len(gad_cols)} GAD columns")

# 5. PHQ (depression) columns
phq_cols = [c for c in df.columns if 'PHQ' in c]
print(f"   Removing {len(phq_cols)} PHQ columns")

# 6. Allodynia columns
allodynia_cols = [c for c in df.columns if 'Allodynia' in c or 'allodynia' in c]
print(f"   Removing {len(allodynia_cols)} Allodynia columns")

# 7. Medication columns
medication_cols = [c for c in df.columns if 'Medication name' in c]
print(f"   Removing {len(medication_cols)} medication columns")

# 8. VAS score
vas_cols = [c for c in df.columns if c == 'VAS']
print(f"   Removing {len(vas_cols)} VAS columns")

# 9. Frequency (post-diagnosis tracking)
freq_cols = [c for c in df.columns if 'Frequency' in c and 'Months' in c]
print(f"   Removing {len(freq_cols)} frequency tracking columns")

# Combine all leakage columns
leakage_cols = (diagnosis_cols + midas_cols + hit6_cols + gad_cols + 
                phq_cols + allodynia_cols + medication_cols + vas_cols + freq_cols)

# Remove "serial number" or ID columns
id_cols = [c for c in df.columns if 'serial' in c.lower() or c == 'introduction_can be']
leakage_cols.extend(id_cols)

print(f"\n📊 Total leakage columns to remove: {len(set(leakage_cols))}")

# Create target BEFORE removing anything
if 'target' in df.columns:
    y = df['target'].values
    print("✅ Target already exists")
else:
    # Create from any diagnosis column that contains "Migraine"
    y = np.zeros(len(df))
    for col in diagnosis_cols:
        migraine_mask = df[col].fillna(0).astype(str).str.contains('Migraine', case=False, na=False)
        y = np.logical_or(y, migraine_mask).astype(int)
    print(f"✅ Created target: {np.sum(y)} migraines, {len(y)-np.sum(y)} non-migraines")

# Drop leakage columns
df_clean = df.drop(columns=set(leakage_cols), errors='ignore')
if 'target' in df_clean.columns:
    df_clean = df_clean.drop(columns=['target'])

print(f"   Remaining columns: {df_clean.shape[1]}")

# ============================================================================
# HANDLE MISSING DATA
# ============================================================================
print("\n🔍 Handling missing data...")

missing_pct = df_clean.isnull().sum() / len(df_clean)
high_missing = missing_pct[missing_pct > 0.7].index.tolist()

if high_missing:
    print(f"   Dropping {len(high_missing)} columns with >70% missing")
    df_clean = df_clean.drop(columns=high_missing)

print(f"   Final feature count: {df_clean.shape[1]}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
X = df_clean

print(f"\n✅ Clean dataset:")
print(f"   Samples: {len(y)}")
print(f"   Features: {X.shape[1]}")
print(f"   Migraines: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
print(f"   Non-migraines: {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")

# ============================================================================
# IDENTIFY COLUMN TYPES
# ============================================================================
print("\n🔢 Identifying feature types...")

numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Convert yes/no to binary
for col in categorical_cols[:]:
    if X[col].astype(str).str.lower().isin(['yes', 'no']).mean() > 0.5:
        X[col] = X[col].astype(str).str.lower().map({'yes': 1, 'no': 0})
        categorical_cols.remove(col)
        numerical_cols.append(col)

print(f"   Numerical: {len(numerical_cols)}")
print(f"   Categorical: {len(categorical_cols)}")

# ============================================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================================
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
], remainder='drop')

# ============================================================================
# SPLIT AND TRANSFORM
# ============================================================================
print("\n✂️  Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("🔄 Transforming...")
X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

print(f"   Processed features: {X_train_clean.shape[1]}")

# ============================================================================
# SAVE
# ============================================================================
print("\n💾 Saving...")

np.save('X_train_CLEAN.npy', X_train_clean)
np.save('X_test_CLEAN.npy', X_test_clean)
np.save('y_train_CLEAN.npy', y_train)
np.save('y_test_CLEAN.npy', y_test)
joblib.dump(preprocessor, 'preprocessor_CLEAN.joblib')

# Save column info
pd.DataFrame({
    'Column': X.columns,
    'Type': ['Numerical' if c in numerical_cols else 'Categorical' for c in X.columns]
}).to_csv('features_CLEAN.csv', index=False)

# Save what was removed
pd.DataFrame({'Removed_Leakage': list(set(leakage_cols))}).to_csv('removed_columns.csv', index=False)

print("\n✅ COMPLETE!")
print("\n📦 Files:")
print("   • X_train_CLEAN.npy, X_test_CLEAN.npy")
print("   • y_train_CLEAN.npy, y_test_CLEAN.npy")
print("   • preprocessor_CLEAN.joblib")
print("   • features_CLEAN.csv (features used)")
print("   • removed_columns.csv (leakage removed)")

print("\n🎯 Expected AUC after fix: 0.75-0.85 (REALISTIC!)")
print("="*80)