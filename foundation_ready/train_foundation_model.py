# compare_models.py
"""
MigraineMamba - Comprehensive Model Comparison
For Research Paper Publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

plt.style.use('seaborn-v0_8-paper')

print("="*80)
print("🧠 MIGRAINEMAMBA - COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("📂 Loading data...")
X_train = np.load("X_train_CLEAN.npy")
X_test = np.load("X_test_CLEAN.npy")
y_train = np.load("y_train_CLEAN.npy")
y_test = np.load("y_test_CLEAN.npy")
print(f"✅ Loaded: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test\n")

# ============================================================================
# DEFINE MODELS
# ============================================================================
models = {
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        random_state=42, verbose=False
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    )
}

print(f"🤖 Models to compare: {list(models.keys())}\n")

# ============================================================================
# TRAIN AND EVALUATE
# ============================================================================
print("⏳ Training all models (5-10 minutes total)...\n")

results = []
trained_models = {}
roc_data = {}

for i, (name, model) in enumerate(models.items(), 1):
    print(f"[{i}/{len(models)}] Training {name}...")
    start = datetime.now()
    
    # Train
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start).total_seconds()
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    metrics = {
        'Model': name,
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'Training_Time': train_time
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['TN'], metrics['FP'] = cm[0, 0], cm[0, 1]
    metrics['FN'], metrics['TP'] = cm[1, 0], cm[1, 1]
    
    results.append(metrics)
    trained_models[name] = model
    
    # ROC data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['AUC-ROC']}
    
    print(f"   ✅ AUC={metrics['AUC-ROC']:.4f}, "
          f"Acc={metrics['Accuracy']:.4f}, Time={train_time:.1f}s\n")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("📊 5-fold cross-validation...\n")

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"   CV {name}...")
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=skf, scoring='roc_auc', n_jobs=-1
    )
    cv_results[name] = cv_scores
    print(f"      Mean={cv_scores.mean():.4f} ±{cv_scores.std():.4f}")

print()

# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================
print("📈 Statistical testing (paired t-tests)...\n")

results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
best_name = results_df.iloc[0]['Model']
best_cv = cv_results[best_name]

print(f"Best model: {best_name} (AUC={results_df.iloc[0]['AUC-ROC']:.4f})\n")

for name, scores in cv_results.items():
    if name != best_name:
        t_stat, p_val = stats.ttest_rel(best_cv, scores)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{name:20s} vs {best_name}: p={p_val:.4f} {sig}")

print("\n   *** p<0.001, ** p<0.01, * p<0.05, ns=not significant\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("💾 Saving results...\n")

results_df.to_csv('model_comparison_results.csv', index=False)
print("✅ model_comparison_results.csv")

pd.DataFrame(cv_results).to_csv('cross_validation_scores.csv', index=False)
print("✅ cross_validation_scores.csv")

best_model = trained_models[best_name]
joblib.dump(best_model, f'best_model_{best_name.replace(" ", "_")}.joblib')
print(f"✅ best_model_{best_name.replace(' ', '_')}.joblib")

# ============================================================================
# VISUALIZATIONS (Publication-Quality)
# ============================================================================
print("\n📊 Creating figures (300 DPI)...\n")

# Figure 1: ROC Curves
print("   Figure 1: ROC curves...")
plt.figure(figsize=(10, 8))
for name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], 
             label=f"{name} (AUC={data['auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ figure1_roc_comparison.png")

# Figure 2: Metrics Comparison
print("   Figure 2: Metrics bar chart...")
fig, ax = plt.subplots(figsize=(12, 6))
metrics_cols = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
results_df.set_index('Model')[metrics_cols].plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figure2_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ figure2_metrics_comparison.png")

# Figure 3: Cross-Validation Box Plot
print("   Figure 3: CV distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
pd.DataFrame(cv_results).boxplot(ax=ax, grid=False)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('AUC-ROC (5-Fold CV)', fontsize=12)
ax.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figure3_cv_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ figure3_cv_distribution.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
report = f"""
{'='*80}
MIGRAINEMAMBA - MODEL COMPARISON REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE RESULTS (Ranked by AUC-ROC)
{'-'*80}
"""

for idx, row in results_df.iterrows():
    report += f"\n{idx+1}. {row['Model']}\n"
    report += f"   AUC-ROC:   {row['AUC-ROC']:.4f}\n"
    report += f"   Accuracy:  {row['Accuracy']:.4f}\n"
    report += f"   Precision: {row['Precision']:.4f}\n"
    report += f"   Recall:    {row['Recall']:.4f}\n"
    report += f"   F1-Score:  {row['F1-Score']:.4f}\n"

report += f"""
CROSS-VALIDATION RESULTS
{'-'*80}
"""
for name, scores in cv_results.items():
    report += f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}\n"

report += f"""
BEST MODEL
{'-'*80}
{best_name}
Test AUC: {results_df.iloc[0]['AUC-ROC']:.4f}
CV AUC: {cv_results[best_name].mean():.4f} ± {cv_results[best_name].std():.4f}

FOR YOUR PAPER
{'-'*80}
"We evaluated five machine learning algorithms (XGBoost, LightGBM, 
CatBoost, Random Forest, and Logistic Regression) using 5-fold 
stratified cross-validation. {best_name} achieved the highest 
performance with AUC-ROC of {results_df.iloc[0]['AUC-ROC']:.4f} on the 
test set and {cv_results[best_name].mean():.4f} ± {cv_results[best_name].std():.4f} 
in cross-validation."
{'='*80}
"""

with open('model_comparison_report.txt', 'w') as f:
    f.write(report)

print(report)
print("✅ model_comparison_report.txt")

print("\n" + "="*80)
print("✅ COMPARISON COMPLETE!")
print("="*80)
print(f"\n🏆 Winner: {best_name} (AUC={results_df.iloc[0]['AUC-ROC']:.4f})")
print("\n📦 Generated Files:")
print("   • model_comparison_results.csv")
print("   • cross_validation_scores.csv")
print(f"   • best_model_{best_name.replace(' ', '_')}.joblib")
print("   • figure1_roc_comparison.png")
print("   • figure2_metrics_comparison.png")
print("   • figure3_cv_distribution.png")
print("   • model_comparison_report.txt")
print("\n" + "="*80)