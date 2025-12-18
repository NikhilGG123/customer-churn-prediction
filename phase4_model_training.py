# Phase 4: Model Training & Evaluation
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
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

print("=" * 80)
print("PHASE 4: MODEL TRAINING & EVALUATION")
print("=" * 80)

# Load preprocessed data
print("\n[1] Loading Data")
print("-" * 80)
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples")
print(f"Churn rate: Train {y_train.mean() * 100:.2f}%, Test {y_test.mean() * 100:.2f}%")

# Train models
print("\n[2] Training Models")
print("-" * 80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

trained_models = {}
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV ROC-AUC': cv_scores.mean()
    })

    print(f"✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# Compare models
print("\n[3] Model Comparison")
print("-" * 80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
best_roc_auc = results_df['ROC-AUC'].max()

print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

results_df.to_csv('model_comparison.csv', index=False)
print("\n✓ Saved: model_comparison.csv")

# Visualize comparison
print("\n[4] Creating Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Accuracy
axes[0, 0].barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0, 0].set_xlabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
axes[0, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Precision vs Recall
axes[0, 1].scatter(results_df['Recall'], results_df['Precision'], s=200, alpha=0.6, c='coral')
for i, model in enumerate(results_df['Model']):
    axes[0, 1].annotate(model, (results_df['Recall'][i], results_df['Precision'][i]), fontsize=9, ha='center')
axes[0, 1].set_xlabel('Recall', fontweight='bold')
axes[0, 1].set_ylabel('Precision', fontweight='bold')
axes[0, 1].set_title('Precision vs Recall', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# F1-Score
axes[1, 0].barh(results_df['Model'], results_df['F1-Score'], color='lightgreen')
axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[1, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['F1-Score']):
    axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

# ROC-AUC
axes[1, 1].barh(results_df['Model'], results_df['ROC-AUC'], color='plum')
axes[1, 1].set_xlabel('ROC-AUC', fontweight='bold')
axes[1, 1].set_title('ROC-AUC Comparison', fontweight='bold')
axes[1, 1].set_xlim([0, 1])
for i, v in enumerate(results_df['ROC-AUC']):
    axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: model_comparison.png")

# Detailed evaluation of best model
print(f"\n[5] Best Model Evaluation: {best_model_name}")
print("-" * 80)

y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives: {tn} | False Positives: {fp}")
print(f"False Negatives: {fn} | True Positives: {tp}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churn']))

# Visualize best model performance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Best Model: {best_model_name}', fontsize=16, fontweight='bold')

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual', fontweight='bold')
axes[0].set_xlabel('Predicted', fontweight='bold')

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {best_roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1].set_xlabel('False Positive Rate', fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontweight='bold')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: best_model_evaluation.png")

# Save model
print("\n[6] Saving Model")
print("-" * 80)
joblib.dump(best_model, 'best_model.pkl')
print(f"✓ Saved: best_model.pkl ({best_model_name})")

# Business impact calculation
print("\n[7] Business Impact")
print("-" * 80)

avg_customer_value = 2000
retention_cost = 100
total_customers = len(y_test)
actual_churners = sum(y_test)
predicted_churners = sum(y_pred_best)

revenue_saved = tp * (avg_customer_value - retention_cost)
wasted_cost = fp * retention_cost
net_benefit = revenue_saved - wasted_cost

print(f"\nTest Set ({total_customers} customers):")
print(f"  Actual churners: {actual_churners} ({actual_churners / total_customers * 100:.1f}%)")
print(f"  Predicted churners: {predicted_churners}")
print(f"  Correctly identified: {tp} ({recall_score(y_test, y_pred_best) * 100:.1f}% recall)")
print(f"  False alarms: {fp}")

print(f"\n  Revenue saved: ${revenue_saved:,}")
print(f"  Wasted costs: ${wasted_cost:,}")
print(f"  Net benefit: ${net_benefit:,}")

annual_projection = net_benefit * (7043 / total_customers)
print(f"\n  Projected annual benefit: ${annual_projection:,.0f}")

# Summary
print("\n" + "=" * 80)
print("PHASE 4 COMPLETE")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"  Accuracy: {results_df[results_df['Model'] == best_model_name]['Accuracy'].values[0]:.3f}")
print(f"  Precision: {results_df[results_df['Model'] == best_model_name]['Precision'].values[0]:.3f}")
print(f"  Recall: {results_df[results_df['Model'] == best_model_name]['Recall'].values[0]:.3f}")
print(f"  ROC-AUC: {best_roc_auc:.3f}")

print(f"\nBusiness Value:")
print(f"  Catches {recall_score(y_test, y_pred_best) * 100:.1f}% of churners")
print(f"  Net benefit: ${net_benefit:,}")
print(f"  Annual projection: ${annual_projection:,.0f}")

print("\n✓ Trained 4 models")
print("✓ Best model saved")
print("✓ All visualizations created")
print("\nNext: Deploy with Streamlit")
print("=" * 80)