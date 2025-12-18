# Phase 3: Feature Engineering & Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 3: FEATURE ENGINEERING & PREPROCESSING")
print("="*80)

df = pd.read_csv('data_checkpoint_phase2.csv')
print(f"\n✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create new features
print("\n[1] Feature Engineering")
print("-" * 80)

# Tenure groups - customers are vulnerable early on
def tenure_group(tenure):
    if tenure <= 6: return '0-6 months'
    elif tenure <= 12: return '6-12 months'
    elif tenure <= 24: return '1-2 years'
    elif tenure <= 48: return '2-4 years'
    else: return '4+ years'

df['TenureGroup'] = df['tenure'].apply(tenure_group)
print("✓ TenureGroup (bucketed tenure)")

# Customer lifetime value
df['CLV'] = df['MonthlyCharges'] * df['tenure']
print("✓ CLV (Customer Lifetime Value)")

# Charges ratio - detects price changes
df['ChargesRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'])
df['ChargesRatio'].fillna(0, inplace=True)
df['ChargesRatio'] = df['ChargesRatio'].replace([np.inf, -np.inf], 0)
print("✓ ChargesRatio (price change indicator)")

# Service adoption score
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df['ServiceCount'] = 0
for col in service_cols:
    df['ServiceCount'] += ((df[col] == 'Yes') | (df[col] == 'DSL') | (df[col] == 'Fiber optic')).astype(int)
print("✓ ServiceCount (# of services)")

# Tech support flag - strong predictor
df['HasTechSupport'] = (df['TechSupport'] == 'Yes').astype(int)
print("✓ HasTechSupport")

# Security services flag
df['HasSecurityServices'] = ((df['OnlineSecurity'] == 'Yes') |
                              (df['OnlineBackup'] == 'Yes') |
                              (df['DeviceProtection'] == 'Yes')).astype(int)
print("✓ HasSecurityServices")

# Charges category
def charges_category(charges):
    if charges < 30: return 'Low'
    elif charges < 70: return 'Medium'
    else: return 'High'

df['ChargesCategory'] = df['MonthlyCharges'].apply(charges_category)
print("✓ ChargesCategory")

print(f"\n✓ Created 7 features | New shape: {df.shape}")

# Encode categorical variables
print("\n[2] Encoding")
print("-" * 80)

df_processed = df.copy()

# Binary encoding
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

# Encode target
df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encoding
categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'Contract', 'PaymentMethod', 'TenureGroup', 'ChargesCategory']

df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# Drop unnecessary columns
df_encoded = df_encoded.drop(columns=['customerID', 'SeniorCitizen_Label'], errors='ignore')

print(f"✓ Encoded categorical features | Shape: {df_encoded.shape}")

# Prepare features and target
print("\n[3] Train-Test Split")
print("-" * 80)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Features: {X.shape[1]} columns")
print(f"Target: {len(y)} samples (No Churn: {sum(y==0)}, Churn: {sum(y==1)})")

# Split with stratification to maintain churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% churn)")
print(f"Test:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% churn)")

# Scale numeric features
print("\n[4] Feature Scaling")
print("-" * 80)

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'ChargesRatio', 'ServiceCount']
cols_to_scale = [col for col in numeric_cols if col in X_train.columns]

scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print(f"✓ Scaled {len(cols_to_scale)} features: {', '.join(cols_to_scale)}")

# Save everything
print("\n[5] Saving Files")
print("-" * 80)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
joblib.dump(scaler, 'scaler.pkl')

with open('feature_names.txt', 'w') as f:
    for feature in X_train.columns:
        f.write(f"{feature}\n")

df_encoded.to_csv('data_preprocessed_full.csv', index=False)

print("✓ X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("✓ scaler.pkl")
print("✓ feature_names.txt")
print("✓ data_preprocessed_full.csv")

# Summary
print("\n" + "="*80)
print("PHASE 3 COMPLETE")
print("="*80)
print(f"\nDataset: {len(df_encoded)} samples")
print(f"Training: {len(X_train)} ({len(X_train)/len(df_encoded)*100:.1f}%)")
print(f"Testing: {len(X_test)} ({len(X_test)/len(df_encoded)*100:.1f}%)")
print(f"Features: {X_train.shape[1]}")
print(f"Churn rate: {y.mean()*100:.2f}%")

print("\nFeatures Engineered:")
print("  1. TenureGroup | 2. CLV | 3. ChargesRatio | 4. ServiceCount")
print("  5. HasTechSupport | 6. HasSecurityServices | 7. ChargesCategory")

print("\n✓ Ready for modeling")
print("\nNext: Phase 4 - Model Training")
print("="*80)