"""
Phase 3: Feature Engineering & Preprocessing
Create predictive features, encode variables, and prepare train/test splits
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("PHASE 3: FEATURE ENGINEERING & PREPROCESSING\n")

df = pd.read_csv('data/data_checkpoint_phase2.csv')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Feature engineering
print("Creating new features:")

# Tenure buckets - early customers are most vulnerable
def tenure_group(tenure):
    if tenure <= 6: return '0-6 months'
    elif tenure <= 12: return '6-12 months'
    elif tenure <= 24: return '1-2 years'
    elif tenure <= 48: return '2-4 years'
    else: return '4+ years'

df['TenureGroup'] = df['tenure'].apply(tenure_group)
print("  TenureGroup (customer lifecycle stage)")

# Customer lifetime value
df['CLV'] = df['MonthlyCharges'] * df['tenure']
print("  CLV (customer lifetime value)")

# Price change indicator
df['ChargesRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'])
df['ChargesRatio'].fillna(0, inplace=True)
df['ChargesRatio'] = df['ChargesRatio'].replace([np.inf, -np.inf], 0)
print("  ChargesRatio (detects pricing changes)")

# Service engagement score
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df['ServiceCount'] = 0
for col in service_cols:
    df['ServiceCount'] += ((df[col] == 'Yes') | (df[col] == 'DSL') | (df[col] == 'Fiber optic')).astype(int)
print("  ServiceCount (engagement intensity)")

# Strong predictor flags
df['HasTechSupport'] = (df['TechSupport'] == 'Yes').astype(int)
df['HasSecurityServices'] = ((df['OnlineSecurity'] == 'Yes') |
                              (df['OnlineBackup'] == 'Yes') |
                              (df['DeviceProtection'] == 'Yes')).astype(int)
print("  HasTechSupport, HasSecurityServices")

# Charges category for price sensitivity
def charges_category(charges):
    if charges < 30: return 'Low'
    elif charges < 70: return 'Medium'
    else: return 'High'

df['ChargesCategory'] = df['MonthlyCharges'].apply(charges_category)
print("  ChargesCategory (price segment)")

print(f"\nCreated 7 features | New shape: {df.shape}")

# Encoding
print("\nEncoding variables:")

df_processed = df.copy()

# Binary encoding for Yes/No features
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encoding for multi-category features
categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'Contract', 'PaymentMethod', 'TenureGroup', 'ChargesCategory']

df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
df_encoded = df_encoded.drop(columns=['customerID', 'SeniorCitizen_Label'], errors='ignore')

print(f"Encoded categorical features | Shape: {df_encoded.shape}")

# Prepare features and target
print("\nPreparing train/test split:")

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Features: {X.shape[1]} columns")
print(f"Samples: {len(y)} (No Churn: {sum(y==0)}, Churn: {sum(y==1)})")

# Stratified split to maintain churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% churn)")
print(f"Test:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% churn)")

# Scale numeric features
print("\nScaling numeric features:")

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'ChargesRatio', 'ServiceCount']
cols_to_scale = [col for col in numeric_cols if col in X_train.columns]

scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print(f"Scaled {len(cols_to_scale)} features: {', '.join(cols_to_scale)}")

# Save all outputs
print("\nSaving files:")

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
joblib.dump(scaler, 'models/scaler.pkl')

with open('models/feature_names.txt', 'w') as f:
    for feature in X_train.columns:
        f.write(f"{feature}\n")

df_encoded.to_csv('data/data_preprocessed_full.csv', index=False)

print("  Train/test CSVs saved to data/")
print("  Scaler and feature names saved to models/")

# Summary
print(f"\nPhase 3 Complete:")
print(f"  Total samples: {len(df_encoded)}")
print(f"  Training: {len(X_train)} ({len(X_train)/len(df_encoded)*100:.1f}%)")
print(f"  Testing: {len(X_test)} ({len(X_test)/len(df_encoded)*100:.1f}%)")
print(f"  Features: {X_train.shape[1]}")
print(f"  Churn rate: {y.mean()*100:.2f}%")

print("\nEngineered features:")
print("  TenureGroup, CLV, ChargesRatio, ServiceCount,")
print("  HasTechSupport, HasSecurityServices, ChargesCategory")

print("\nNext: Phase 4 - Model Training")