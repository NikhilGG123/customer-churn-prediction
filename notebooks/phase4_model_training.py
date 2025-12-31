"""
Phase 4: Model Training & Evaluation
Train multiple ML models, compare performance, and calculate business impact
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.model_selection import cross_val_score
import joblib
import warnings

warnings.filterwarnings('ignore')

print("PHASE 4: MODEL TRAINING & EVALUATION\n")

# Load preprocessed data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples")
print(f"Churn rate: {y_train.mean() * 100:.2f}% (train), {y_test.mean() * 100:.2f}% (test)\n")

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

print("Training models:")
trained_models = {}
results = []

for name, model in models.items():
    print(f"\n{name}...")

    # Train model
    model.fit(X_train, y_train)
    trained_models[name] = model

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'CV ROC-AUC': cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    }
    results.append(metrics)

    print(
        f"  Accuracy: {metrics['Accuracy']:.4f} | Recall: {metrics['Recall']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")

# Compare models
print("\n\nModel Comparison:")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
best_roc_auc = results_df['ROC-AUC'].max()

print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

results_df.to_csv('data/model_comparison.csv', index=False)

# Create comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

axes[0, 0].barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0, 0].set_xlabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
axes[0, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

axes[0, 1].scatter(results_df['Recall'], results_df['Precision'], s=200, alpha=0.6, c='coral')
for i, model in enumerate(results_df['Model']):
    axes[0, 1].annotate(model, (results_df['Recall'][i], results_df['Precision'][i]), fontsize=9, ha='center')
axes[0, 1].set_xlabel('Recall', fontweight='bold')
axes[0, 1].set_ylabel('Precision', fontweight='bold')
axes[0, 1].set_title('Precision vs Recall', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].barh(results_df['Model'], results_df['F1-Score'], color='lightgreen')
axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[1, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['F1-Score']):
    axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

axes[1, 1].barh(results_df['Model'], results_df['ROC-AUC'], color='plum')
axes[1, 1].set_xlabel('ROC-AUC', fontweight='bold')
axes[1, 1].set_title('ROC-AUC Comparison', fontweight='bold')
axes[1, 1].set_xlim([0, 1])
for i, v in enumerate(results_df['ROC-AUC']):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: visualizations/model_comparison.png")

# Detailed evaluation of best model
print(f"\nEvaluating {best_model_name}:")

y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(cm)
print(f"  True Negatives: {tn} | False Positives: {fp}")
print(f"  False Negatives: {fn} | True Positives: {tp}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churn']))

# Visualize best model performance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Best Model: {best_model_name}', fontsize=16, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual', fontweight='bold')
axes[0].set_xlabel('Predicted', fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {best_roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1].set_xlabel('False Positive Rate', fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontweight='bold')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/best_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/best_model_evaluation.png")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\nSaved: models/best_model.pkl ({best_model_name})")

# Calculate business impact
print("\nBusiness Impact Analysis:")

avg_customer_value = 2000
retention_cost = 100

revenue_saved = tp * (avg_customer_value - retention_cost)
wasted_cost = fp * retention_cost
net_benefit = revenue_saved - wasted_cost

print(f"\nTest Set ({len(y_test)} customers):")
print(f"  Actual churners: {sum(y_test)} ({sum(y_test) / len(y_test) * 100:.1f}%)")
print(f"  Correctly identified: {tp} ({recall_score(y_test, y_pred_best) * 100:.1f}% recall)")
print(f"  False alarms: {fp}")

print(f"\n  Revenue saved: ${revenue_saved:,}")
print(f"  Wasted costs: ${wasted_cost:,}")
print(f"  Net benefit: ${net_benefit:,}")

annual_projection = net_benefit * (7043 / len(y_test))
print(f"  Projected annual: ${annual_projection:,.0f}")

# Summary
print(f"\nPhase 4 Complete:")
print(f"  Best model: {best_model_name}")
print(f"  ROC-AUC: {best_roc_auc:.3f}")
print(f"  Business value: ${net_benefit:,} (test set)")
print(f"  Annual projection: ${annual_projection:,.0f}")

print("\nNext: Deploy with Streamlit (streamlit_app.py)")