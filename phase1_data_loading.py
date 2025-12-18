# Phase 1: Data Loading & Initial EDA
# Customer Churn Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("CUSTOMER CHURN PREDICTION - PHASE 1: DATA LOADING & INITIAL EDA")
print("=" * 80)

# ============================================================================
# 1. LOAD THE DATA
# ============================================================================
print("\n[1] Loading Data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"✓ Data loaded successfully!")
print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# 2. INITIAL DATA INSPECTION
# ============================================================================
print("\n[2] Initial Data Inspection")
print("-" * 80)
print("\nFirst 5 rows:")
print(df.head())

print("\n\nColumn Names and Data Types:")
print(df.dtypes)

print("\n\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 3. CHECK FOR MISSING VALUES
# ============================================================================
print("\n[3] Missing Values Analysis")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print("\nColumns with missing values:")
    print(missing_df)
else:
    print("✓ No missing values found!")

# ============================================================================
# 4. TARGET VARIABLE ANALYSIS (CHURN)
# ============================================================================
print("\n[4] Target Variable Analysis: Churn")
print("-" * 80)

# Check if TotalCharges has any issues (it's often stored as object instead of numeric)
print(f"\nTotalCharges data type: {df['TotalCharges'].dtype}")
if df['TotalCharges'].dtype == 'object':
    print("⚠ TotalCharges is stored as object (string) - will need to convert to numeric")
    # Check for non-numeric values
    non_numeric = df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]['TotalCharges']
    print(f"  Found {len(non_numeric)} non-numeric values in TotalCharges")

# Churn distribution
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print("\nChurn Distribution:")
print(f"  No:  {churn_counts['No']} ({churn_pct['No']:.2f}%)")
print(f"  Yes: {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")
print(f"\n  Churn Rate: {churn_pct['Yes']:.2f}%")

# Visualize churn distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for i, v in enumerate(churn_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Pie chart
colors = ['#2ecc71', '#e74c3c']
axes[1].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Churn Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'churn_distribution.png'")
plt.show()

# ============================================================================
# 5. FEATURE CATEGORIES
# ============================================================================
print("\n[5] Feature Categorization")
print("-" * 80)

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove customerID and Churn from categorical
categorical_features.remove('customerID')
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')

# Remove TotalCharges from numeric if it's there (we'll handle it separately)
if 'TotalCharges' in categorical_features:
    categorical_features.remove('TotalCharges')

print(f"\nNumeric Features ({len(numeric_features)}):")
print(f"  {', '.join(numeric_features)}")

print(f"\nCategorical Features ({len(categorical_features)}):")
print(f"  {', '.join(categorical_features)}")

# ============================================================================
# 6. SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1 COMPLETE - KEY FINDINGS:")
print("=" * 80)
print(f"✓ Dataset has {df.shape[0]} customers and {df.shape[1]} features")
print(f"✓ Churn rate: {churn_pct['Yes']:.2f}% (imbalanced - we'll handle this)")
print(f"✓ TotalCharges needs cleaning (object → numeric conversion)")
print(f"✓ Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
print("\nNext Step: Phase 2 - Deep EDA and Visualizations")
print("=" * 80)

# Save the dataframe for next phase
df.to_csv('data_checkpoint_phase1.csv', index=False)
print("\n✓ Data checkpoint saved as 'data_checkpoint_phase1.csv'")
