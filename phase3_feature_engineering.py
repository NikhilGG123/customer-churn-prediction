# Phase 3: Feature Engineering & Preprocessing
# Customer Churn Prediction Project

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CUSTOMER CHURN PREDICTION - PHASE 3: FEATURE ENGINEERING & PREPROCESSING")
print("="*80)

# Load data
df = pd.read_csv('data_checkpoint_phase2.csv')
print(f"\n✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# 1. CREATE NEW FEATURES (FEATURE ENGINEERING)
# ============================================================================
print("\n[1] Feature Engineering - Creating New Features")
print("-" * 80)

# Feature 1: Tenure Groups (customers are most vulnerable in first 12 months)
def tenure_group(tenure):
    if tenure <= 6:
        return '0-6 months'
    elif tenure <= 12:
        return '6-12 months'
    elif tenure <= 24:
        return '1-2 years'
    elif tenure <= 48:
        return '2-4 years'
    else:
        return '4+ years'

df['TenureGroup'] = df['tenure'].apply(tenure_group)
print("✓ Created: TenureGroup (0-6 months, 6-12 months, 1-2 years, 2-4 years, 4+ years)")

# Feature 2: Customer Lifetime Value (CLV)
df['CLV'] = df['MonthlyCharges'] * df['tenure']
print("✓ Created: CLV (Customer Lifetime Value = MonthlyCharges × tenure)")

# Feature 3: Charges Ratio (detects if customer got price increase/decrease)
df['ChargesRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'])
df['ChargesRatio'].fillna(0, inplace=True)  # Handle division by zero for tenure=0
df['ChargesRatio'] = df['ChargesRatio'].replace([np.inf, -np.inf], 0)
print("✓ Created: ChargesRatio (TotalCharges / (MonthlyCharges × tenure))")

# Feature 4: Service Adoption Score (how many services they use)
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df['ServiceCount'] = 0
for col in service_cols:
    df['ServiceCount'] += (df[col] == 'Yes').astype(int) | (df[col] == 'DSL').astype(int) | (df[col] == 'Fiber optic').astype(int)

print("✓ Created: ServiceCount (number of services customer uses)")

# Feature 5: Has Tech Support (binary feature - tech support is a strong predictor)
df['HasTechSupport'] = (df['TechSupport'] == 'Yes').astype(int)
print("✓ Created: HasTechSupport (binary: 0=No, 1=Yes)")

# Feature 6: Has Security Services (combines online security, backup, device protection)
df['HasSecurityServices'] = ((df['OnlineSecurity'] == 'Yes') |
                              (df['OnlineBackup'] == 'Yes') |
                              (df['DeviceProtection'] == 'Yes')).astype(int)
print("✓ Created: HasSecurityServices (has any security-related service)")

# Feature 7: Monthly Charges Category
def charges_category(charges):
    if charges < 30:
        return 'Low'
    elif charges < 70:
        return 'Medium'
    else:
        return 'High'

df['ChargesCategory'] = df['MonthlyCharges'].apply(charges_category)
print("✓ Created: ChargesCategory (Low < $30, Medium $30-70, High > $70)")

print(f"\n✓ Total features created: 7")
print(f"  New dataframe shape: {df.shape}")

# ============================================================================
# 2. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n[2] Encoding Categorical Variables")
print("-" * 80)

# Create a copy for processing
df_processed = df.copy()

# Binary encoding for Yes/No columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
print(f"✓ Binary encoded: {', '.join(binary_cols)}")

# Encode target variable
df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
print("✓ Target variable 'Churn' encoded: Yes=1, No=0")

# One-hot encoding for multi-category features
categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'Contract', 'PaymentMethod', 'TenureGroup', 'ChargesCategory']

df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
print(f"✓ One-hot encoded {len(categorical_cols)} categorical features")
print(f"  New shape after encoding: {df_encoded.shape}")

# Drop unnecessary columns
cols_to_drop = ['customerID', 'SeniorCitizen_Label']
df_encoded = df_encoded.drop(columns=cols_to_drop, errors='ignore')
print(f"✓ Dropped unnecessary columns: {', '.join(cols_to_drop)}")

# ============================================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[3] Preparing Features and Target Variable")
print("-" * 80)

# Separate features (X) and target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"✓ Features (X): {X.shape[1]} columns")
print(f"✓ Target (y): {y.shape[0]} samples")
print(f"  - Class distribution: No Churn={sum(y==0)}, Churn={sum(y==1)}")

# Show feature names
print(f"\nFeature columns ({len(X.columns)}):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4] Splitting Data into Train and Test Sets")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train.shape[0]} samples ({len(y_train[y_train==1])} churned)")
print(f"✓ Test set:  {X_test.shape[0]} samples ({len(y_test[y_test==1])} churned)")
print(f"  Train churn rate: {y_train.mean()*100:.2f}%")
print(f"  Test churn rate:  {y_test.mean()*100:.2f}%")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================
print("\n[5] Scaling Numeric Features")
print("-" * 80)

# Identify numeric columns that need scaling
numeric_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV',
                         'ChargesRatio', 'ServiceCount']

# Only scale columns that exist in X_train
cols_to_scale = [col for col in numeric_cols_to_scale if col in X_train.columns]

# Create and fit scaler on training data only
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print(f"✓ Scaled {len(cols_to_scale)} numeric features using StandardScaler")
print(f"  Features scaled: {', '.join(cols_to_scale)}")

# ============================================================================
# 6. SAVE PREPROCESSED DATA
# ============================================================================
print("\n[6] Saving Preprocessed Data and Objects")
print("-" * 80)

# Save train/test splits
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("✓ Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# Save scaler for future use
joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

# Save feature names
feature_names = X_train.columns.tolist()
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")
print("✓ Saved: feature_names.txt")

# Save full processed dataframe
df_encoded.to_csv('data_preprocessed_full.csv', index=False)
print("✓ Saved: data_preprocessed_full.csv")

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("PHASE 3 COMPLETE - PREPROCESSING SUMMARY")
print("="*80)

print(f"\n📊 DATASET STATISTICS:")
print(f"  • Total samples: {len(df_encoded)}")
print(f"  • Training samples: {len(X_train)} ({len(X_train)/len(df_encoded)*100:.1f}%)")
print(f"  • Test samples: {len(X_test)} ({len(X_test)/len(df_encoded)*100:.1f}%)")
print(f"  • Total features: {X_train.shape[1]}")

print(f"\n🎯 TARGET DISTRIBUTION:")
print(f"  • Overall churn rate: {y.mean()*100:.2f}%")
print(f"  • Training churn rate: {y_train.mean()*100:.2f}%")
print(f"  • Test churn rate: {y_test.mean()*100:.2f}%")

print(f"\n🔧 FEATURES ENGINEERED:")
print(f"  1. TenureGroup - Customer tenure buckets")
print(f"  2. CLV - Customer Lifetime Value")
print(f"  3. ChargesRatio - Price change detection")
print(f"  4. ServiceCount - Number of services used")
print(f"  5. HasTechSupport - Tech support indicator")
print(f"  6. HasSecurityServices - Security services indicator")
print(f"  7. ChargesCategory - Monthly charges category")

print(f"\n✓ All data preprocessed and ready for modeling!")
print("✓ Files saved successfully")
print("\nNext Step: Phase 4 - Model Training & Evaluation")
print("="*80)