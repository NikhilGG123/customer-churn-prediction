# Phase 4: Model Training & Evaluation (Fixed - No Interactive Plots)
# Customer Churn Prediction Project

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
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
print("CUSTOMER CHURN PREDICTION - PHASE 4: MODEL TRAINING & EVALUATION")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] Loading Preprocessed Data")
print("-" * 80)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"✓ Training churn rate: {y_train.mean() * 100:.2f}%")
print(f"✓ Test churn rate: {y_test.mean() * 100:.2f}%")

# ============================================================================
# 2. TRAIN MULTIPLE MODELS
# ============================================================================
print("\n[2] Training Multiple Models")
print("-" * 80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

# Dictionary to store trained models and results
trained_models = {}
results = []

for name, model in models.items():
    print(f"\n⏳ Training {name}...")

    # Train the model
    model.fit(X_train, y_train)
    trained_models[name] = model

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation score (on training data)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    cv_mean = cv_scores.mean()

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV ROC-AUC': cv_mean
    })

    print(f"✓ {name} trained successfully")
    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# 3. MODEL COMPARISON
# ============================================================================
print("\n[3] Model Comparison")
print("-" * 80)

results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# Find best model based on ROC-AUC
best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
best_roc_auc = results_df['ROC-AUC'].max()

print(f"\n🏆 BEST MODEL: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

# Save results
results_df.to_csv('model_comparison.csv', index=False)
print("\n✓ Saved: model_comparison.csv")

# ============================================================================
# 4. VISUALIZE MODEL COMPARISON
# ============================================================================
print("\n[4] Creating Model Comparison Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
axes[0, 0].barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0, 0].set_xlabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
axes[0, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Plot 2: Precision vs Recall
axes[0, 1].scatter(results_df['Recall'], results_df['Precision'], s=200, alpha=0.6, c='coral')
for i, model in enumerate(results_df['Model']):
    axes[0, 1].annotate(model, (results_df['Recall'][i], results_df['Precision'][i]),
                        fontsize=9, ha='center')
axes[0, 1].set_xlabel('Recall', fontweight='bold')
axes[0, 1].set_ylabel('Precision', fontweight='bold')
axes[0, 1].set_title('Precision vs Recall', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_ylim([0, 1])

# Plot 3: F1-Score Comparison
axes[1, 0].barh(results_df['Model'], results_df['F1-Score'], color='lightgreen')
axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[1, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['F1-Score']):
    axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Plot 4: ROC-AUC Comparison
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

# ============================================================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ============================================================================
print(f"\n[5] Detailed Evaluation of Best Model: {best_model_name}")
print("-" * 80)

# Make predictions with best model
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)
print(f"\nBreakdown:")
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives (TN): {tn} - Correctly predicted No Churn")
print(f"  False Positives (FP): {fp} - Incorrectly predicted Churn")
print(f"  False Negatives (FN): {fn} - Missed actual churners")
print(f"  True Positives (TP): {tp} - Correctly predicted Churn")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churn']))

# ============================================================================
# 6. VISUALIZE CONFUSION MATRIX AND ROC CURVE
# ============================================================================
print("\n[6] Creating Detailed Visualizations for Best Model")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Detailed Performance: {best_model_name}', fontsize=16, fontweight='bold')

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Actual', fontweight='bold')
axes[0].set_xlabel('Predicted', fontweight='bold')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {best_roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontweight='bold')
axes[1].set_title('ROC Curve', fontweight='bold', fontsize=12)
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: best_model_evaluation.png")

# ============================================================================
# 7. SAVE BEST MODEL
# ============================================================================
print("\n[7] Saving Best Model")
print("-" * 80)

joblib.dump(best_model, 'best_model.pkl')
print(f"✓ Saved best model ({best_model_name}) as: best_model.pkl")

# ============================================================================
# 8. BUSINESS IMPACT CALCULATION
# ============================================================================
print("\n[8] Business Impact Analysis")
print("-" * 80)

# Assumptions
avg_customer_value = 2000  # Average customer lifetime value
retention_cost = 100  # Cost to retain a customer
total_test_customers = len(y_test)
actual_churners = sum(y_test)

# Model predictions
predicted_churners = sum(y_pred_best)
true_positives = sum((y_test == 1) & (y_pred_best == 1))
false_positives = sum((y_test == 0) & (y_pred_best == 1))

# Calculate savings
revenue_saved = true_positives * (avg_customer_value - retention_cost)
wasted_cost = false_positives * retention_cost
net_benefit = revenue_saved - wasted_cost

print(f"\n💰 BUSINESS IMPACT METRICS:")
print(f"  Test Set Size: {total_test_customers} customers")
print(f"  Actual Churners: {actual_churners} ({actual_churners / total_test_customers * 100:.1f}%)")
print(f"  Predicted Churners: {predicted_churners}")
print(f"  Correctly Identified Churners (TP): {true_positives} ({recall_score(y_test, y_pred_best) * 100:.1f}% recall)")
print(f"  False Alarms (FP): {false_positives}")
print(f"\n  Revenue Saved: ${revenue_saved:,.0f}")
print(f"  Wasted Retention Costs: ${wasted_cost:,.0f}")
print(f"  Net Benefit: ${net_benefit:,.0f}")

# Annualized projection (if we apply this to full customer base)
annual_projection = net_benefit * (7043 / total_test_customers)
print(f"\n  📊 Projected Annual Benefit (full dataset): ${annual_projection:,.0f}")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4 COMPLETE - MODEL TRAINING SUMMARY")
print("=" * 80)

print(f"\n🎯 BEST MODEL: {best_model_name}")
print(f"  • Accuracy: {results_df[results_df['Model'] == best_model_name]['Accuracy'].values[0]:.3f}")
print(f"  • Precision: {results_df[results_df['Model'] == best_model_name]['Precision'].values[0]:.3f}")
print(f"  • Recall: {results_df[results_df['Model'] == best_model_name]['Recall'].values[0]:.3f}")
print(f"  • F1-Score: {results_df[results_df['Model'] == best_model_name]['F1-Score'].values[0]:.3f}")
print(f"  • ROC-AUC: {best_roc_auc:.3f}")

print(f"\n💼 BUSINESS VALUE:")
print(f"  • Catches {recall_score(y_test, y_pred_best) * 100:.1f}% of churners")
print(f"  • Net benefit: ${net_benefit:,.0f} on test set")
print(f"  • Projected annual savings: ${annual_projection:,.0f}")

print(f"\n✓ Trained and evaluated 4 models")
print(f"✓ Best model saved as: best_model.pkl")
print(f"✓ All visualizations and results saved")

print("\nNext Step: Phase 5 - Build Streamlit App for Deployment")
print("=" * 80)