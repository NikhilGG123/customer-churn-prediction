"""
Phase 1: Data Loading & Initial Exploratory Data Analysis
Load dataset, inspect data quality, and analyze churn distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure display and plot settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("PHASE 1: DATA LOADING & INITIAL EXPLORATION\n")

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns\n")

# Display first rows and data types
print("First 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found")
else:
    print(missing[missing > 0])

# Identify TotalCharges data type issue
print("\nData Quality Check:")
if df['TotalCharges'].dtype == 'object':
    print("TotalCharges is stored as string - requires conversion")
    non_numeric = df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]['TotalCharges']
    print(f"Found {len(non_numeric)} non-numeric values\n")

# Analyze target variable distribution
print("Churn Distribution:")
churn_counts = df['Churn'].value_counts()
churn_pct = churn_counts / len(df) * 100
print(f"No:  {churn_counts['No']} ({churn_pct['No']:.2f}%)")
print(f"Yes: {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")

# Visualize churn distribution
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
plt.savefig('visualizations/churn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: visualizations/churn_distribution.png")

# Categorize features by type
print("\nFeature Types:")
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove ID and target from categoricals
categorical_features.remove('customerID')
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')
if 'TotalCharges' in categorical_features:
    categorical_features.remove('TotalCharges')

print(f"Numeric ({len(numeric_features)}): {', '.join(numeric_features)}")
print(f"Categorical ({len(categorical_features)}): {', '.join(categorical_features)}")

# Summary
print(f"\nPhase 1 Complete:")
print(f"  Dataset: {df.shape[0]} customers, {df.shape[1]} features")
print(f"  Churn rate: {churn_pct['Yes']:.2f}%")
print(f"  Issue identified: TotalCharges needs type conversion")
print("\nNext: Phase 2 - Deep EDA")

# Save checkpoint
df.to_csv('data/data_checkpoint_phase1.csv', index=False)
print("\nCheckpoint saved: data/data_checkpoint_phase1.csv")