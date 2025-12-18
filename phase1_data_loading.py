# Phase 1: Data Loading & Initial EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("PHASE 1: DATA LOADING & INITIAL EXPLORATION")
print("="*80)

# Load dataset
print("\n[1] Loading Data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"✓ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# Initial inspection
print("\n[2] Dataset Overview")
print("-" * 80)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\n[3] Missing Values")
print("-" * 80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found")
else:
    print(missing[missing > 0])

# Analyze target variable
print("\n[4] Target Variable: Churn")
print("-" * 80)

# Check TotalCharges data type issue
if df['TotalCharges'].dtype == 'object':
    print("⚠ TotalCharges is stored as string - needs conversion")
    non_numeric = df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]['TotalCharges']
    print(f"  Found {len(non_numeric)} non-numeric values")

# Churn distribution
churn_counts = df['Churn'].value_counts()
churn_pct = churn_counts / len(df) * 100

print(f"\nChurn Distribution:")
print(f"  No:  {churn_counts['No']} ({churn_pct['No']:.2f}%)")
print(f"  Yes: {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")

# Visualize churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
for i, v in enumerate(churn_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

axes[1].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Churn Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: churn_distribution.png")
plt.show()

# Categorize features
print("\n[5] Feature Types")
print("-" * 80)
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove ID and target from categoricals
categorical_features.remove('customerID')
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')
if 'TotalCharges' in categorical_features:
    categorical_features.remove('TotalCharges')

print(f"\nNumeric features ({len(numeric_features)}): {', '.join(numeric_features)}")
print(f"\nCategorical features ({len(categorical_features)}): {', '.join(categorical_features)}")

# Summary
print("\n" + "="*80)
print("PHASE 1 COMPLETE")
print("="*80)
print(f"✓ Dataset: {df.shape[0]} customers, {df.shape[1]} features")
print(f"✓ Churn rate: {churn_pct['Yes']:.2f}%")
print(f"✓ Issue identified: TotalCharges needs type conversion")
print("\nNext: Phase 2 - Deep EDA")
print("="*80)

# Save checkpoint
df.to_csv('data_checkpoint_phase1.csv', index=False)
print("\n✓ Checkpoint saved: data_checkpoint_phase1.csv")